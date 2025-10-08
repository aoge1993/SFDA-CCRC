import torch
import numpy as np
import medpy.metric.binary as medmetric

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(1,2))
    # same for the ground truth
    # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(1,2))
    # same for the intersection
    intersection = np.sum(intersection.astype(float), axis=(1,2))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value

def accuracy1(binary_segmentation, binary_gt_label):
    '''
    Compute the accuracy between two binary segmentation.
    Accuracy is defined as the ratio of correctly classified pixels to the total number of pixels.
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        accuracy_value: Accuracy between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

    # compute the total number of pixels
    total_pixels = binary_segmentation.size

    # compute the number of correctly classified pixels (True Positives and True Negatives)
    correct_pixels = np.sum(binary_segmentation == binary_gt_label)

    # compute the accuracy
    accuracy_value = correct_pixels / total_pixels

    # return it
    return accuracy_value

def dice_coefficient_numpy_3D(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 3D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 3D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(0,1,2))
    # same for the ground truth
    # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(0,1,2))
    # same for the intersection
    intersection = np.sum(intersection.astype(float), axis=(0,1,2))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_numpy_medpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    return medmetric.dc(binary_segmentation, binary_gt_label)


    # if get_hd:
    #     if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
    #         return medmetric.assd(binary_segmentation, binary_gt_label)
    #         # return medmetric.hd(binary_segmentation, binary_gt_label)
    #     else:
    #         return np.nan
    # else:
    #     return 0.0


def assd_numpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
        return medmetric.assd(binary_segmentation, binary_gt_label)
    else:
        return -1


def hd_numpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation)
    binary_gt_label = np.asarray(binary_gt_label)

    if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
        return medmetric.hd(binary_segmentation, binary_gt_label)
    else:
        return -1


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    return dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), dice_coefficient_numpy(pred[:, 1, ...],
                                                                                              target[:, 1, ...])


def dice_coeff_5label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # target = target.data.cpu()
    # pred = torch.sigmoid(pred)
    # pred = pred.data.cpu()
    # pred[pred > 0.75] = 1
    # pred[pred <= 0.75] = 0
    # return (dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]),
    #         dice_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...]),
    #         dice_coefficient_numpy(pred[:, 2, ...], target[:, 2, ...]),
    #         dice_coefficient_numpy(pred[:, 3, ...], target[:, 3, ...]),
    #         dice_coefficient_numpy(pred[:, 4, ...], target[:, 4, ...]))
    y_hat = torch.argmax(pred, dim=1, keepdim=True)
    pred_label = torch.zeros(pred.size())  # bs*4*W*H  bs*5*W*H
    if torch.cuda.is_available():
        pred_label = pred_label.cuda()
    pred_label = pred_label.scatter_(1, y_hat, 1)  # one-hot label
    target = target.data.cpu()
    pred_label = pred_label.data.cpu()
    return (dice_coefficient_numpy(pred_label[:, 0, ...], target[:, 0, ...]),
            dice_coefficient_numpy(pred_label[:, 1, ...], target[:, 1, ...]),
            dice_coefficient_numpy(pred_label[:, 2, ...], target[:, 2, ...]),
            dice_coefficient_numpy(pred_label[:, 3, ...], target[:, 3, ...]),
            dice_coefficient_numpy(pred_label[:, 4, ...], target[:, 4, ...]))
def dice_coeff_4label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    y_hat = torch.argmax(pred, dim=1, keepdim=True)
    pred_label = torch.zeros(pred.size())  # bs*4*W*H
    if torch.cuda.is_available():
        pred_label = pred_label.cuda()
    pred_label = pred_label.scatter_(1, y_hat, 1)  # one-hot label
    target = target.data.cpu()
    pred_label = pred_label.data.cpu()
    return (dice_coefficient_numpy(pred_label[:, 0, ...], target[:, 0, ...]),
            dice_coefficient_numpy(pred_label[:, 1, ...], target[:, 1, ...]),
            dice_coefficient_numpy(pred_label[:, 2, ...], target[:, 2, ...]),
            dice_coefficient_numpy(pred_label[:, 3, ...], target[:, 3, ...]))
def acc_5label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    y_hat = torch.argmax(pred, dim=1, keepdim=True)
    pred_label = torch.zeros(pred.size())  # bs*4*W*H  bs*5*W*H
    if torch.cuda.is_available():
        pred_label = pred_label.cuda()
    pred_label = pred_label.scatter_(1, y_hat, 1)  # one-hot label
    target = target.data.cpu()
    pred_label = pred_label.data.cpu()
    return (accuracy1(pred_label[:, 0, ...], target[:, 0, ...]),
            accuracy1(pred_label[:, 1, ...], target[:, 1, ...]),
            accuracy1(pred_label[:, 2, ...], target[:, 2, ...]),
            accuracy1(pred_label[:, 3, ...], target[:, 3, ...]),
            accuracy1(pred_label[:, 4, ...], target[:, 4, ...]))
def acc_5label2(pred, target):
    """Calculate the accuracy for 5-label segmentation, including per-class accuracy.
    pred: tensor with shape (batch_size, num_classes, height, width)
    target: tensor with shape (batch_size, num_classes, height, width)  (one-hot encoded)
    Returns:
        overall_accuracy: Overall accuracy across all classes.
        class_accuracy: A list of accuracy values for each class.
    """
    # Get the predicted labels (class indices)
    y_hat = torch.argmax(pred, dim=1)  # shape: (batch_size, height, width)
    device = pred.device
    target = target.to(device)  # 将 target 移动到 GPU


    # Convert target from one-hot to class indices
    target_indices = torch.argmax(target, dim=1)  # shape: (batch_size, height, width)

    # Calculate overall accuracy
    correct_pixels = (y_hat == target_indices).sum().item()
    total_pixels = target_indices.numel()
    overall_accuracy = correct_pixels / total_pixels

    # print("ooooo0",overall_accuracy)

    # Calculate per-class accuracy
    num_classes = pred.size(1)  # Number of classes
    class_accuracy = []

    for class_idx in range(num_classes):
        # Mask for the current class
        class_mask = (target_indices == class_idx)

        # Total pixels for the current class
        total_class_pixels = class_mask.sum().item()

        if total_class_pixels == 0:
            # If the class has no pixels, set accuracy to 0 or skip
            class_accuracy.append(0.0)
            continue

        # Correctly predicted pixels for the current class
        correct_class_pixels = ((y_hat == class_idx) & (target_indices == class_idx)).sum().item()

        # Accuracy for the current class
        class_accuracy.append(correct_class_pixels / total_class_pixels)
    # print("ioio",class_accuracy[0],class_accuracy[1],class_accuracy[2],class_accuracy[3],class_accuracy[4])
    return overall_accuracy, class_accuracy
def DiceLoss(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def assd_compute(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0

    assd = np.zeros([pred.shape[0], pred.shape[1]])
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            assd[i][j] = assd_numpy(pred[i, j, ...], target[i, j, ...])

    return assd
