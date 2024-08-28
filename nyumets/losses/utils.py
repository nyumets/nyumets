from monai.losses import (
    DiceLoss,
    DiceFocalLoss,
    FocalLoss,
    GeneralizedDiceLoss,
    GeneralizedDiceFocalLoss,
    DiceCELoss,
    TverskyLoss
)

def get_loss_function(loss_function: str = 'dice'):
    if loss_function.lower() == 'dice':
        loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_function.lower() == 'dice_focal':
        loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_function.lower() == 'focal':
        loss_function = FocalLoss(include_background=False, to_onehot_y=True)
    elif loss_function.lower() == 'generalized_dice':
        loss_function = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_function.lower() == 'generalized_dice_focal':
        loss_function = GeneralizedDiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_function.lower() == 'dice_ce':
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    elif loss_function.lower() == 'tversky':
        loss_function = TverskyLoss(include_background=False, to_onehot_y=True, softmax=True)
    return loss_function
