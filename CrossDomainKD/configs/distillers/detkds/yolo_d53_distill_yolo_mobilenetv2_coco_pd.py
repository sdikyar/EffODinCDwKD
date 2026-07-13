_base_ = [
    '../../yolo/yolov3_mobilenetv2_mstrain-416_300e_coco_base.py',
    '../../_base_/datasets/coco_detection_yolo_pd.py',
]
work_dir = './work_dirs/retina_detkds_yolo'
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_Yolo',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth',
    init_student = True,
    distill_cfg = [
    # Scale 1: Deepest/Coarsest features (Stride 32) -> Corresponds to FPN P5
    dict(
        student_module='neck.detect1',
        teacher_module='neck.detect1',
        output_hook=True,
        methods=[dict(
            type='FeatureLoss',
            name='loss_yolo_s32',
            student_channels=96,  # Matches YOLOv3Neck out_channels
            teacher_channels=96,  # Update this if teacher is not YOLOv3-96
            global_trans1='catt',
            global_trans2='no',
            global_trans3='no',
            global_dis='pear',
            gb_enable=True,
            fbg_trans1='satt',
            fbg_trans2='no',
            fbg_trans3='no',
            fbg_dis='pear',
            fbg_enable=True,
            logits_trans='no',
            logits_dis='kl',
            logits_enable=True,
            gamma_global=0.5,
            gamma_fbg=4,
            gamma_logits=1
        )]
    ),
    # Scale 2: Medium features (Stride 16) -> Corresponds to FPN P4
    dict(
        student_module='neck.detect2',
        teacher_module='neck.detect2',
        output_hook=True,
        methods=[dict(
            type='FeatureLoss',
            name='loss_yolo_s16',
            student_channels=96,
            teacher_channels=96,
            global_trans1='catt',
            global_trans2='no',
            global_trans3='no',
            global_dis='pear',
            gb_enable=True,
            fbg_trans1='satt',
            fbg_trans2='no',
            fbg_trans3='no',
            fbg_dis='pear',
            fbg_enable=True,
            logits_trans='no',
            logits_dis='kl',
            logits_enable=True,
            gamma_global=0.5,
            gamma_fbg=4,
            gamma_logits=1
        )]
    ),
    # Scale 3: Finest features (Stride 8) -> Corresponds to FPN P3
    dict(
        student_module='neck.detect3',
        teacher_module='neck.detect3',
        output_hook=True,
        methods=[dict(
            type='FeatureLoss',
            name='loss_yolo_s8',
            student_channels=96,
            teacher_channels=96,
            global_trans1='catt',
            global_trans2='no',
            global_trans3='no',
            global_dis='pear',
            gb_enable=True,
            fbg_trans1='satt',
            fbg_trans2='no',
            fbg_trans3='no',
            fbg_dis='pear',
            fbg_enable=True,
            logits_trans='no',
            logits_dis='kl',
            logits_enable=True,
            gamma_global=0.5,
            gamma_fbg=4,
            gamma_logits=1
        )]
    )
]
    )
student_cfg = 'configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py'
teacher_cfg = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.0001,
    step=[24, 28])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
evaluation = dict(interval=1, metric=['bbox'])
