_base_ = [
    '../../_base_/datasets/coco_detection_pd8.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_GFL_Cross',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
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
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/gfl/gfl_r18_fpn_1x_coco.py'
teacher_cfg = 'configs/gfl/gfl_r50_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=24)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)