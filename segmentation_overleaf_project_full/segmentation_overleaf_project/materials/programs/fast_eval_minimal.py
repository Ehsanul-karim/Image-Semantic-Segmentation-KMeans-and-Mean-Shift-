#!/usr/bin/env python3
from pathlib import Path
import sys, argparse, re, time, csv
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, '/mnt/data/work_input')
import evaluate_2 as ev

IMG_DIR_RE = re.compile(r'^Img\d+$', re.I)
POST_RE = re.compile(r'^post\s*processed', re.I)

def labels_from_image(path: Path) -> Tuple[np.ndarray, List[str]]:
    arr = np.asarray(Image.open(path).convert('RGB'))
    rgb = arr.astype(np.uint32)
    packed = (rgb[:,:,0]<<16) + (rgb[:,:,1]<<8) + rgb[:,:,2]
    unique, inv = np.unique(packed.reshape(-1), return_inverse=True)
    names = [f'#{int((v>>16)&255):02x}{int((v>>8)&255):02x}{int(v&255):02x}' for v in unique]
    return inv.reshape(packed.shape).astype(np.int32), names

def resize_labels(labels: np.ndarray, shape: Tuple[int,int]) -> np.ndarray:
    if labels.shape == shape: return labels.astype(np.int32, copy=False)
    im = Image.fromarray(labels.astype(np.int32), mode='I')
    im = im.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(im).astype(np.int32)

def compact(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique, inv = np.unique(labels.reshape(-1), return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int32), unique.astype(np.int32)

def contingency(gt: np.ndarray, pred: np.ndarray, n_gt=None, n_pred=None) -> np.ndarray:
    if n_gt is None: n_gt=int(gt.max())+1
    if n_pred is None: n_pred=int(pred.max())+1
    idx = gt.reshape(-1).astype(np.int64)*n_pred + pred.reshape(-1).astype(np.int64)
    return np.bincount(idx, minlength=n_gt*n_pred).reshape(n_gt,n_pred).astype(np.int64)

def boundary_map(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask): return np.zeros(mask.shape, dtype=bool)
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3,3), dtype=bool), border_value=0)

def boundary_f1_from_maps(gt_b: np.ndarray, gt_dil: np.ndarray, pred_mask: np.ndarray, tol:int=2) -> float:
    pred_b = boundary_map(pred_mask)
    n_gt=int(gt_b.sum()); n_pred=int(pred_b.sum())
    if n_gt==0 and n_pred==0: return 1.0
    if n_gt==0 or n_pred==0: return 0.0
    pred_dil = ndi.binary_dilation(pred_b, iterations=tol)
    precision = float((pred_b & gt_dil).sum())/n_pred if n_pred else 0.0
    recall = float((gt_b & pred_dil).sum())/n_gt if n_gt else 0.0
    return (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0

def discover(root: Path, gt_folder: str):
    items = ev.discover_predictions(root, gt_folder)
    return items

def safe_div(a,b): return float(a/b) if b else 0.0

def eval_root(root: Path, gt_folder: str, out_dir: str, perf_dir: str, boundary_tol=2, gt_source='mask'):
    t_all=time.time()
    out=root/out_dir; perf=root/perf_dir
    out.mkdir(parents=True, exist_ok=True); perf.mkdir(parents=True, exist_ok=True)
    items=discover(root, gt_folder)
    print('predictions', len(items), flush=True)
    # Preload GT once per image
    gt_cache={}
    for image_id in sorted(set(x.image_id for x in items)):
        any_item=next(x for x in items if x.image_id==image_id)
        gt=ev.load_gt_data(image_id, any_item.gt_mask_path, any_item.gt_json_path, gt_source, True)
        n_gt=len(gt.class_names)
        gt_bound=[]; gt_dils=[]
        for cid in range(n_gt):
            b=boundary_map(gt.labels==cid)
            gt_bound.append(b)
            gt_dils.append(ndi.binary_dilation(b, iterations=boundary_tol))
        gt_cache[image_id]=(gt, gt_bound, gt_dils)
    per_image=[]; per_class=[]; cm_long=[]; label_matches=[]
    for i,item in enumerate(items,1):
        if i==1 or i%50==0:
            print(i, '/', len(items), item.rel_path, flush=True)
        gt, gt_bound, gt_dils = gt_cache[item.image_id]
        pred, pred_names = labels_from_image(item.pred_path)
        pred = resize_labels(pred, gt.labels.shape)
        pred, old_ids = compact(pred)
        pred_names_resized=[pred_names[int(o)] if int(o)<len(pred_names) else str(int(o)) for o in old_ids]
        n_gt=len(gt.class_names); n_pred=len(pred_names_resized)
        cm0=contingency(gt.labels, pred, n_gt, n_pred)
        # many-to-one mapping: each predicted label maps to GT label with maximum overlap.
        pred_to_gt=np.argmax(cm0, axis=0).astype(np.int32) if n_pred else np.array([], dtype=np.int32)
        remapped=pred_to_gt[pred] if n_pred else np.zeros_like(pred)
        cm=contingency(gt.labels, remapped, n_gt, n_gt)
        total=int(cm.sum()); acc=sum(int(cm[k,k]) for k in range(n_gt))
        precs=[]; recs=[]; ious=[]; bfs=[]; wp=wr=wi=wb=0.0; support_total=0
        base={
            'image_id': item.image_id,
            'feature_set': item.feature_set,
            'method_dir': item.method_dir,
            'algorithm': item.algorithm,
            'prediction_name': item.prediction_name,
            'method_key': item.method_key,
            'gt_source_type': gt.source_type,
            'gt_source_path': str(gt.source_path),
            'gt_mask_path': str(item.gt_mask_path) if item.gt_mask_path else '',
            'gt_json_path': str(item.gt_json_path) if item.gt_json_path else '',
            'prediction_path': str(item.pred_path),
            'prediction_rel_path': item.rel_path,
            'matching': 'many_to_one',
            'gt_shape': f'{gt.labels.shape[0]}x{gt.labels.shape[1]}',
            'n_gt_labels': n_gt,
            'n_pred_labels': n_pred,
            'boundary_tolerance_px': boundary_tol,
            'include_background_in_macro': False,
        }
        for cid in range(n_gt):
            tp=int(cm[cid,cid]); fp=int(cm[:,cid].sum()-tp); fn=int(cm[cid,:].sum()-tp); support=int(cm[cid,:].sum())
            pr=safe_div(tp,tp+fp); re=safe_div(tp,tp+fn); io=safe_div(tp,tp+fp+fn)
            bf=boundary_f1_from_maps(gt_bound[cid], gt_dils[cid], remapped==cid, boundary_tol)
            include=(cid!=0)
            if include:
                precs.append(pr); recs.append(re); ious.append(io); bfs.append(bf)
                wp += support*pr; wr += support*re; wi += support*io; wb += support*bf; support_total += support
            per_class.append({**base,'class_id':cid,'class_name':gt.class_names[cid], 'support_pixels':support,
                              'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':re,'iou':io,'boundary_f1':bf,
                              'included_in_macro_average':include})
        row={**base,
             'pixel_accuracy': safe_div(acc,total),
             'macro_precision': float(np.mean(precs)) if precs else np.nan,
             'macro_recall': float(np.mean(recs)) if recs else np.nan,
             'macro_iou': float(np.mean(ious)) if ious else np.nan,
             'macro_boundary_f1': float(np.mean(bfs)) if bfs else np.nan,
             'weighted_precision': safe_div(wp, support_total),
             'weighted_recall': safe_div(wr, support_total),
             'weighted_iou': safe_div(wi, support_total),
             'weighted_boundary_f1': safe_div(wb, support_total)}
        per_image.append(row)
        row_sums=cm.sum(axis=1, keepdims=True)
        cm_norm=np.divide(cm,row_sums,out=np.zeros_like(cm,dtype=float),where=row_sums!=0)
        for r in range(n_gt):
            for c in range(n_gt):
                cm_long.append({**base,'gt_label_id':r,'gt_label_name':gt.class_names[r],
                                'pred_label_after_matching_id':c,'pred_label_after_matching_name':gt.class_names[c],
                                'raw_count':int(cm[r,c]),'row_normalized_value':float(cm_norm[r,c])})
        for pid, gid in enumerate(pred_to_gt.tolist()):
            label_matches.append({**base,'pred_label_id':pid,'pred_label_value':pred_names_resized[pid],
                                  'matched_gt_label_id':gid,'matched_gt_label_value':gt.class_names[gid],
                                  'overlap_pixels':int(cm0[gid,pid]),'is_extra_prediction_label':False})
    per_image_df=pd.DataFrame(per_image); per_class_df=pd.DataFrame(per_class); cm_df=pd.DataFrame(cm_long); lm_df=pd.DataFrame(label_matches)
    per_image_df.to_csv(out/'per_image_results.csv', index=False)
    per_class_df.to_csv(out/'per_class_results.csv', index=False)
    cm_df.to_csv(out/'confusion_matrices_long.csv', index=False)
    lm_df.to_csv(out/'label_matches.csv', index=False)
    # summaries
    def save_group(df, cols, name):
        g=(df.groupby(cols, as_index=False)
            .agg(n_results=('image_id','size'), n_images=('image_id','nunique'),
                 macro_precision=('macro_precision','mean'), macro_recall=('macro_recall','mean'),
                 macro_iou=('macro_iou','mean'), macro_boundary_f1=('macro_boundary_f1','mean'),
                 pixel_accuracy=('pixel_accuracy','mean'), weighted_iou=('weighted_iou','mean'))
            .sort_values(['macro_iou','macro_boundary_f1','macro_precision'], ascending=False))
        g.to_csv(perf/name, index=False); return g
    macro_by_method=save_group(per_image_df,['method_key'],'macro_by_method.csv')
    macro_by_feature_algorithm=save_group(per_image_df,['feature_set','algorithm'],'macro_by_feature_algorithm.csv')
    macro_by_algorithm=save_group(per_image_df,['algorithm'],'macro_by_algorithm.csv')
    overall=pd.DataFrame([{'n_results':len(per_image_df),'n_images':per_image_df.image_id.nunique(),
        'macro_precision':per_image_df.macro_precision.mean(),'macro_recall':per_image_df.macro_recall.mean(),
        'macro_iou':per_image_df.macro_iou.mean(),'macro_boundary_f1':per_image_df.macro_boundary_f1.mean(),
        'pixel_accuracy':per_image_df.pixel_accuracy.mean(),'weighted_iou':per_image_df.weighted_iou.mean()}])
    overall.to_csv(perf/'overall_macro_average.csv', index=False)
    per_image_df.to_csv(perf/'per_image_results.csv', index=False); per_class_df.to_csv(perf/'per_class_results.csv', index=False)
    print('done seconds', time.time()-t_all, flush=True)
    print(macro_by_method[['method_key','n_results','n_images','macro_precision','macro_recall','macro_iou','macro_boundary_f1']].head(15).to_string(index=False), flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gt-folder', default='GT Ehsanul Karim')
    ap.add_argument('--out-dir', default='output_fast')
    ap.add_argument('--perf-dir', default='performance_fast')
    ap.add_argument('--gt-source', default='mask')
    args=ap.parse_args()
    eval_root(Path(args.root).resolve(), args.gt_folder, args.out_dir, args.perf_dir, gt_source=args.gt_source)
