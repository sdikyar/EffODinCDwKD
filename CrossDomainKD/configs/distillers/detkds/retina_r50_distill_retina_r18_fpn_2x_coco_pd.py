
_base_ = [
    '../../_base_/datasets/coco_detection_pd.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
work_dir = './work_dirs/retina_detkds_PD_GFL'
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_Retina_GFL',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='satt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_HW',
                                       global_dis='l2',
                                       gb_enable=True,
                                       fbg_trans1='mask',
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       fbg_enable=True,
                                       logits_trans='softmax_C',
                                       logits_dis='kl',
                                       logits_enable=True,
                                       gamma_global=0.5,
                                       gamma_fbg=5,
                                       gamma_logits=1
                                    )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='satt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_HW',
                                       global_dis='l2',
                                       gb_enable=True,
                                       fbg_trans1='mask',
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       fbg_enable=True,
                                       logits_trans='softmax_C',
                                       logits_dis='kl',
                                       logits_enable=True,
                                       gamma_global=0.5,
                                       gamma_fbg=5,
                                       gamma_logits=1
                                    )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='satt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_HW',
                                       global_dis='l2',
                                       gb_enable=True,
                                       fbg_trans1='mask',
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       fbg_enable=True,
                                       logits_trans='softmax_C',
                                       logits_dis='kl',
                                       logits_enable=True,
                                       gamma_global=0.5,
                                       gamma_fbg=5,
                                       gamma_logits=1
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='satt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_HW',
                                       global_dis='l2',
                                       gb_enable=True,
                                       fbg_trans1='mask',
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       fbg_enable=True,
                                       logits_trans='softmax_C',
                                       logits_dis='kl',
                                       logits_enable=True,
                                       gamma_global=0.5,
                                       gamma_fbg=5,
                                       gamma_logits=1
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='satt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_HW',
                                       global_dis='l2',
                                       gb_enable=True,
                                       fbg_trans1='mask',
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       fbg_enable=True,
                                       logits_trans='softmax_C',
                                       logits_dis='kl',
                                       logits_enable=True,
                                       gamma_global=0.5,
                                       gamma_fbg=5,
                                       gamma_logits=1
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r18_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,)
runner = dict(type='EpochBasedRunner', max_epochs=20)
