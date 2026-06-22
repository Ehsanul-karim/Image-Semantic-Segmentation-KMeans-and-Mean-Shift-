from pathlib import Path
import sys, re
import numpy as np
import pandas as pd
sys.path.insert(0, '/mnt/data')
from fast_eval_minimal import labels_from_image, resize_labels, compact, contingency, boundary_map, boundary_f1_from_maps, safe_div
sys.path.insert(0, '/mnt/data/work_input')
import evaluate_2 as ev
from scipy import ndimage as ndi

root=Path('/mnt/data/work_input/outputs')
own=pd.read_csv(root/'output_fast'/'per_image_results.csv')
# Select the strongest available prediction per image according to own GT macro IoU.
selected=(own.sort_values(['image_id','macro_iou','macro_boundary_f1','macro_precision'], ascending=[True,False,False,False])
          .groupby('image_id', as_index=False).head(1).sort_values('image_id'))
rows=[]; cls_rows=[]; cm_rows=[]
for _,sel in selected.iterrows():
    image_id=sel['image_id']
    pred_path=Path(sel['prediction_path'])
    gt_mask=root/'Peer GT'/image_id/'mask.png'
    gt_json=root/'Peer GT'/image_id/f'{image_id}.json'
    gt=ev.load_gt_data(image_id, gt_mask, gt_json, 'mask', True)
    n_gt=len(gt.class_names)
    gt_bound=[]; gt_dils=[]
    for cid in range(n_gt):
        b=boundary_map(gt.labels==cid); gt_bound.append(b); gt_dils.append(ndi.binary_dilation(b, iterations=2))
    pred, pred_names=labels_from_image(pred_path)
    pred=resize_labels(pred, gt.labels.shape)
    pred, old=compact(pred)
    pred_names_resized=[pred_names[int(o)] if int(o)<len(pred_names) else str(int(o)) for o in old]
    n_pred=len(pred_names_resized)
    cm0=contingency(gt.labels,pred,n_gt,n_pred)
    pred_to_gt=np.argmax(cm0,axis=0).astype(np.int32) if n_pred else np.array([], dtype=np.int32)
    remapped=pred_to_gt[pred] if n_pred else np.zeros_like(pred)
    cm=contingency(gt.labels,remapped,n_gt,n_gt)
    total=int(cm.sum()); acc=sum(int(cm[k,k]) for k in range(n_gt))
    precs=[]; recs=[]; ious=[]; bfs=[]; wp=wr=wi=wb=0.0; support_total=0
    base={'image_id':image_id,'selected_from_own_gt_macro_iou':sel['macro_iou'],
          'selected_from_own_gt_boundary_f1':sel['macro_boundary_f1'],
          'feature_set':sel['feature_set'],'method_dir':sel['method_dir'],'algorithm':sel['algorithm'],
          'prediction_name':sel['prediction_name'],'method_key':sel['method_key'],
          'peer_gt_source_path':str(gt.source_path),'prediction_path':str(pred_path),
          'prediction_rel_path':sel['prediction_rel_path'],'matching':'many_to_one',
          'gt_shape':f'{gt.labels.shape[0]}x{gt.labels.shape[1]}','n_gt_labels':n_gt,'n_pred_labels':n_pred}
    for cid in range(n_gt):
        tp=int(cm[cid,cid]); fp=int(cm[:,cid].sum()-tp); fn=int(cm[cid,:].sum()-tp); support=int(cm[cid,:].sum())
        pr=safe_div(tp,tp+fp); re=safe_div(tp,tp+fn); io=safe_div(tp,tp+fp+fn)
        bf=boundary_f1_from_maps(gt_bound[cid],gt_dils[cid],remapped==cid,2)
        include=cid!=0
        if include:
            precs.append(pr); recs.append(re); ious.append(io); bfs.append(bf)
            wp+=support*pr; wr+=support*re; wi+=support*io; wb+=support*bf; support_total+=support
        cls_rows.append({**base,'class_id':cid,'class_name':gt.class_names[cid], 'support_pixels':support,
                         'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':re,'iou':io,'boundary_f1':bf,
                         'included_in_macro_average':include})
    row={**base,'pixel_accuracy':safe_div(acc,total),'macro_precision':float(np.mean(precs)) if precs else np.nan,
         'macro_recall':float(np.mean(recs)) if recs else np.nan,'macro_iou':float(np.mean(ious)) if ious else np.nan,
         'macro_boundary_f1':float(np.mean(bfs)) if bfs else np.nan,'weighted_precision':safe_div(wp,support_total),
         'weighted_recall':safe_div(wr,support_total),'weighted_iou':safe_div(wi,support_total),
         'weighted_boundary_f1':safe_div(wb,support_total)}
    rows.append(row)
    row_sums=cm.sum(axis=1,keepdims=True); cm_norm=np.divide(cm,row_sums,out=np.zeros_like(cm,dtype=float),where=row_sums!=0)
    for r in range(n_gt):
        for c in range(n_gt):
            cm_rows.append({**base,'gt_label_id':r,'gt_label_name':gt.class_names[r],
                            'pred_label_after_matching_id':c,'pred_label_after_matching_name':gt.class_names[c],
                            'raw_count':int(cm[r,c]),'row_normalized_value':float(cm_norm[r,c])})

out=root/'peer_evaluation'; out.mkdir(exist_ok=True)
per=pd.DataFrame(rows); cls=pd.DataFrame(cls_rows); cm=pd.DataFrame(cm_rows)
per.to_csv(out/'peer_best_per_image_results.csv', index=False)
cls.to_csv(out/'peer_best_per_class_results.csv', index=False)
cm.to_csv(out/'peer_best_confusion_matrices_long.csv', index=False)
overall=pd.DataFrame([{'n_images':per.image_id.nunique(),'n_results':len(per),
    'macro_precision':per.macro_precision.mean(),'macro_recall':per.macro_recall.mean(),
    'macro_iou':per.macro_iou.mean(),'macro_boundary_f1':per.macro_boundary_f1.mean(),
    'pixel_accuracy':per.pixel_accuracy.mean(),'weighted_iou':per.weighted_iou.mean()}])
overall.to_csv(out/'peer_best_overall_summary.csv',index=False)
print(per[['image_id','feature_set','algorithm','method_dir','macro_precision','macro_recall','macro_iou','macro_boundary_f1']].to_string(index=False))
print(overall.to_string(index=False))
