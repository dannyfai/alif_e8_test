#!/usr/bin/env python3
"""Leave-one-worker-out TCN CV, reusing the already-installed testdev script.

All three folds are derived from split_byid/test because no train/val data exist.
This is exploratory only, not an independent benchmark. Original project assets
are read-only; fold checkpoints and reports go into a separate subdirectory.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score


def load_base(script_path: Path):
    spec = importlib.util.spec_from_file_location("tcn_testdev_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper training script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detail_eval(base, project_run, tcn, ref, cla, loader, device):
    tcn.eval(); ref.eval(); cla.eval()
    p = {k: [] for k in ("y_rpe", "y_psi", "ref_rpe", "ref_psi", "tcn_rpe", "tcn_psi")}
    with torch.no_grad():
        for corrupted, _target, _mask, marks, worker, labels in loader:
            x = corrupted.to(device); marks = marks.to(device); worker = worker.to(device)
            labels = labels.numpy()
            missing = (x == -9997) | (x == -9998) | (x == -9999)
            rp = ref(x, x_mark_enc=marks, one_hot_x=worker[:, :18], device=device)
            tp = tcn(x, x_mark_enc=marks, one_hot_x=worker[:, :18], device=device)
            rr = torch.where(missing, rp, x); tr = torch.where(missing, tp, x)
            r_rpe, r_psi = cla(rr, x_mark_enc=marks, one_hot_x=worker[:, :30], device=device)
            t_rpe, t_psi = cla(tr, x_mark_enc=marks, one_hot_x=worker[:, :30], device=device)
            p["y_rpe"].extend(labels[:, 0].tolist()); p["y_psi"].extend(labels[:, 1].tolist())
            p["ref_rpe"].extend(r_rpe.argmax(1).cpu().tolist()); p["ref_psi"].extend(r_psi.argmax(1).cpu().tolist())
            p["tcn_rpe"].extend(t_rpe.argmax(1).cpu().tolist()); p["tcn_psi"].extend(t_psi.argmax(1).cpu().tolist())
    def metric(y, pred):
        y=np.asarray(y,dtype=np.int64); pred=np.asarray(pred,dtype=np.int64); valid=y>=0
        if not valid.any(): return {"n":0,"accuracy":None,"weighted_f1":None}
        return {"n":int(valid.sum()),"accuracy":float(accuracy_score(y[valid],pred[valid])),
                "weighted_f1":float(f1_score(y[valid],pred[valid],average="weighted",zero_division=0))}
    metrics = {
        "reference_reg_rpe":metric(p["y_rpe"],p["ref_rpe"]),
        "tcn_reg_rpe":metric(p["y_rpe"],p["tcn_rpe"]),
        "reference_reg_psi":metric(p["y_psi"],p["ref_psi"]),
        "tcn_reg_psi":metric(p["y_psi"],p["tcn_psi"]),
        "rpe_prediction_agreement":float(np.mean(np.asarray(p["ref_rpe"])==np.asarray(p["tcn_rpe"]))),
        "psi_prediction_agreement":float(np.mean(np.asarray(p["ref_psi"])==np.asarray(p["tcn_psi"]))),
    }
    return metrics,p


def train_fold(args, base, project_run, dataset, fold_i, held_out, all_groups,
               reg_ckpt, cla_ckpt, out_root):
    seed=args.seed + 1009*fold_i
    base.set_seed(seed)
    val_i=[i for i,g in enumerate(dataset.group_ids) if g==held_out]
    train_i=[i for i,g in enumerate(dataset.group_ids) if g!=held_out]
    train_groups=sorted(set(dataset.group_ids[i] for i in train_i))
    if not train_i or not val_i or held_out in train_groups:
        raise RuntimeError(f"Invalid/leaking fold {fold_i}")
    gh=hashlib.sha256(str(held_out).encode()).hexdigest()[:12]
    fold_dir=out_root/f"fold_{fold_i:02d}_heldout_{gh}"; fold_dir.mkdir(parents=True,exist_ok=True)
    train_loader=DataLoader(Subset(dataset,train_i),batch_size=args.batch_size,shuffle=True,
                            num_workers=args.workers,drop_last=False)
    val_loader=DataLoader(Subset(dataset,val_i),batch_size=args.batch_size,shuffle=False,
                          num_workers=args.workers,drop_last=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref=project_run.Reg_Model().cpu(); ref.load_state_dict(base.load_checkpoint(reg_ckpt),strict=True)
    cla=project_run.Cla_Model().cpu(); cla.load_state_dict(base.load_checkpoint(cla_ckpt),strict=True)
    tcn=base.TCNRegModel(project_run,args.channels,args.kernel_size).cpu()
    orig=base.load_checkpoint(reg_ckpt); ts=tcn.state_dict()
    compatible={k:v for k,v in orig.items() if k in ts and ts[k].shape==v.shape}
    missing,unexpected=tcn.load_state_dict(compatible,strict=False)
    if any(k.startswith("rnn.") for k in tcn.state_dict()): raise RuntimeError("Unexpected recurrent weights")
    dataset.set_epoch(0)
    probe=dataset[train_i[0]]; x,_,_,mark,worker,_=probe
    with torch.no_grad():
        rp=ref(x[None],x_mark_enc=mark[None],one_hot_x=worker[:18][None],device=torch.device("cpu"))
        tp=tcn(x[None],x_mark_enc=mark[None],one_hot_x=worker[:18][None],device=torch.device("cpu"))
    expected=(1,base.SEQ_LEN,len(base.SEQ_COLS))
    if tuple(rp.shape)!=expected or tuple(tp.shape)!=expected or not torch.isfinite(tp).all():
        raise RuntimeError(f"Fold {fold_i} CPU forward test failed")
    tcn.to(device); ref.to(device).eval(); cla.to(device).eval()
    scale=torch.tensor(base.LOSS_SCALE,dtype=torch.float32,device=device)
    tcn_params=list(tcn.tcn_in.parameters())+list(tcn.tcn_in_bn.parameters())
    tcn_params+=list(tcn.tcn_blocks.parameters())+list(tcn.tcn_out.parameters())
    ids={id(v) for v in tcn_params}; shared=[v for v in tcn.parameters() if id(v) not in ids]
    opt=torch.optim.AdamW([{"params":tcn_params,"lr":args.lr_tcn},
                           {"params":shared,"lr":args.lr_shared}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=3,min_lr=1e-6)
    best=math.inf; best_epoch=0; stale=0; history=[]
    best_path=fold_dir/"best_tcn_reg.pth"; last_path=fold_dir/"last_tcn_reg.pth"
    conf={"fold":fold_i,"seed":seed,"held_out_worker_hash":gh,
          "train_worker_hashes":[hashlib.sha256(g.encode()).hexdigest()[:12] for g in train_groups],
          "train_samples":len(train_i),"validation_samples":len(val_i),
          "channels":args.channels,"kernel_size":args.kernel_size,"dilations":[1,2]*5,
          "receptive_field":1+2*(args.kernel_size-1)*sum([1,2]*5),
          "warm_started_shared_tensors":len(compatible),"new_tcn_tensors":len(missing),
          "warning":"split_byid/test-derived exploratory CV; not an independent benchmark"}
    (fold_dir/"fold_config.json").write_text(json.dumps(conf,indent=2))
    print(f"\n===== FOLD {fold_i}/{len(all_groups)}: heldout worker hash={gh} =====")
    print(f"Train={len(train_i)} samples/{len(train_groups)} workers; validation={len(val_i)} samples/1 worker")
    print(f"Warm-started shared tensors={len(compatible)}; new TCN tensors={len(missing)}; unexpected={len(unexpected)}")
    for epoch in range(1,args.epochs+1):
        start=time.time(); dataset.set_epoch(epoch); tcn.train(); losses=[]
        for corrupted,target,artificial,marks,worker,_labels in train_loader:
            corrupted=corrupted.to(device); target=target.to(device); artificial=artificial.to(device)
            marks=marks.to(device); worker=worker.to(device); opt.zero_grad(set_to_none=True)
            pred=tcn(corrupted,x_mark_enc=marks,one_hot_x=worker[:,:18],device=device)
            loss=base.masked_reconstruction_loss(pred,target,artificial,scale)
            if not torch.isfinite(loss): raise FloatingPointError(f"Fold {fold_i} epoch {epoch}: non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(tcn.parameters(),1.0); opt.step()
            losses.append(float(loss.item()))
        dataset.set_epoch(0); vl=base.evaluate_reconstruction(tcn,val_loader,device,scale); sched.step(vl)
        tl=float(np.mean(losses)); row={"epoch":epoch,"train_masked_smooth_l1":tl,"val_masked_smooth_l1":vl,
          "lr_tcn":opt.param_groups[0]["lr"],"lr_shared":opt.param_groups[1]["lr"],"seconds":time.time()-start}
        history.append(row); print(f"fold={fold_i} epoch={epoch:03d}/{args.epochs} train={tl:.6f} val={vl:.6f} "
          f"lr_tcn={row['lr_tcn']:.2e} lr_shared={row['lr_shared']:.2e} time={row['seconds']:.1f}s",flush=True)
        payload={"format":"heat_stress_tcn_reg_lowo_cv_v1","model_state_dict":tcn.state_dict(),
                 "config":conf,"epoch":epoch,"train_masked_smooth_l1":tl,"val_masked_smooth_l1":vl}
        torch.save(payload,last_path)
        if vl<best: best=vl; best_epoch=epoch; stale=0; torch.save(payload,best_path)
        else: stale+=1
        pd.DataFrame(history).to_csv(fold_dir/"training_history.csv",index=False)
        if stale>=args.patience:
            print(f"Fold {fold_i}: early stopping after {args.patience} epochs without improvement"); break
    best_payload=torch.load(best_path,map_location="cpu",weights_only=False)
    tcn.load_state_dict(best_payload["model_state_dict"],strict=True); tcn.to(device).eval(); dataset.set_epoch(0)
    metrics,preds=detail_eval(base,project_run,tcn,ref,cla,val_loader,device)
    result={"fold":fold_i,"held_out_worker_hash":gh,"train_samples":len(train_i),"validation_samples":len(val_i),
      "best_epoch":best_epoch,"best_validation_masked_smooth_l1":best,"metrics":metrics,
      "best_checkpoint":str(best_path),"predictions":preds}
    (fold_dir/"validation_results.json").write_text(json.dumps(result,indent=2))
    print("FOLD METRICS:"); print(json.dumps({k:v for k,v in result.items() if k!="predictions"},indent=2))
    return result


def main():
    ap=argparse.ArgumentParser(description="Three-fold leave-one-worker-out TCN validation (exploratory only)")
    ap.add_argument("--project",type=Path,required=True); ap.add_argument("--group-column",default="worker_id")
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--workers",type=int,default=0); ap.add_argument("--channels",type=int,default=32)
    ap.add_argument("--kernel-size",type=int,default=31); ap.add_argument("--lr-tcn",type=float,default=8e-4)
    ap.add_argument("--lr-shared",type=float,default=2e-4); ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--seed",type=int,default=20261121); args=ap.parse_args()
    project=args.project.resolve(); exp=project/"experiments"/"tcn_reg_20261120"
    helper=exp/"train_tcn_reg_testdev_experiment.py"
    if not helper.is_file(): raise FileNotFoundError(f"Missing helper script: {helper}")
    base=load_base(helper)
    data=project/"data"/"split_byid"; models=project/"models"/"trained_by_data_split_byid"
    reg_ckpt=models/"reg_model.pth"; cla_ckpt=models/"rpe_psi_cla_model.pth"
    for p in (project/"run.py",data/"test_seq.feather",data/"test_non_seq.csv",reg_ckpt,cla_ckpt):
        if not p.is_file(): raise FileNotFoundError(f"Required input missing: {p}")
    project_run=base.import_project(project)
    out_root=exp/"leave_one_worker_out_20261121"; out_root.mkdir(parents=True,exist_ok=True)
    print("="*78); print("HEAT-STRESS TCN LEAVE-ONE-WORKER-OUT CROSS-VALIDATION")
    print("DATA SOURCE: split_byid/test ONLY — EXPLORATORY, NOT INDEPENDENT TESTING")
    print(f"Project: {project}\nOutput: {out_root}"); print("Original source/checkpoints/data/firmware are read-only inputs."); print("="*78)
    base.set_seed(args.seed); dataset=base.ImputationDataset(project_run,data,"test",args.seed,args.group_column)
    groups=sorted(set(dataset.group_ids))
    if len(groups)<2: raise RuntimeError(f"Need >=2 worker groups, found {len(groups)}")
    print(f"Group column: {dataset.group_column}; unique workers: {len(groups)}; complete samples: {len(dataset)}")
    folds=[]
    for i,g in enumerate(groups,start=1):
        folds.append(train_fold(args,base,project_run,dataset,i,g,groups,reg_ckpt,cla_ckpt,out_root))
    pooled={k:[] for k in ("y_rpe","y_psi","ref_rpe","ref_psi","tcn_rpe","tcn_psi")}
    for f in folds:
        for k in pooled: pooled[k].extend(f["predictions"][k])
    def met(y,p):
        y=np.asarray(y,dtype=np.int64); p=np.asarray(p,dtype=np.int64); ok=y>=0
        if not ok.any(): return {"n":0,"accuracy":None,"weighted_f1":None}
        return {"n":int(ok.sum()),"accuracy":float(accuracy_score(y[ok],p[ok])),
                "weighted_f1":float(f1_score(y[ok],p[ok],average="weighted",zero_division=0))}
    pooled_metrics={"reference_reg_rpe":met(pooled["y_rpe"],pooled["ref_rpe"]),
      "tcn_reg_rpe":met(pooled["y_rpe"],pooled["tcn_rpe"]),
      "reference_reg_psi":met(pooled["y_psi"],pooled["ref_psi"]),
      "tcn_reg_psi":met(pooled["y_psi"],pooled["tcn_psi"]),
      "rpe_prediction_agreement":float(np.mean(np.asarray(pooled["ref_rpe"])==np.asarray(pooled["tcn_rpe"]))),
      "psi_prediction_agreement":float(np.mean(np.asarray(pooled["ref_psi"])==np.asarray(pooled["tcn_psi"])))}
    summary={"method":"leave-one-worker-out grouped cross-validation","data_source":"split_byid/test",
      "group_column":dataset.group_column,"worker_count":len(groups),"sample_count":len(dataset),
      "channels":args.channels,"kernel_size":args.kernel_size,"dilations":[1,2]*5,
      "theoretical_receptive_field":1+2*(args.kernel_size-1)*sum([1,2]*5),
      "folds":[{k:v for k,v in f.items() if k!="predictions"} for f in folds],
      "pooled_out_of_fold_metrics":pooled_metrics,
      "warning":"All folds derive from split_byid/test. Exploratory CV only; not an independent final benchmark or unbiased estimate of original training-distribution performance.",
      "export_status":"This script does not export, quantize, compile with Vela, or program firmware."}
    out=out_root/"leave_one_worker_out_summary.json"; out.write_text(json.dumps(summary,indent=2))
    (out_root/"pooled_out_of_fold_predictions.json").write_text(json.dumps(pooled))
    print("\n===== POOLED OUT-OF-FOLD SUMMARY ====="); print(json.dumps(summary,indent=2)); print(f"Summary saved: {out}")
    print("Original checkpoints and firmware were not modified.")
    return 0

if __name__=="__main__": raise SystemExit(main())
