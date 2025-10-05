import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    # def _generate_matrix(self, gt_image, pre_image):
    #     mask = (gt_image >= 0) & (gt_image < self.num_class)
    #     label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
    #     count = np.bincount(label, minlength=self.num_class ** 2)
    #     confusion_matrix = count.reshape(self.num_class, self.num_class)
    #     return confusion_matrix

    def _generate_matrix(self, gt_image, pre_image):
    # 确定实际类别数
        actual_classes = max(np.max(gt_image).astype(int) + 1, 
                            np.max(pre_image).astype(int) + 1)
        
        if actual_classes > self.num_class:
            print(f"Warning: Detected {actual_classes} classes, but model uses {self.num_class}")
            self.num_class = actual_classes  # 更新类别数
        
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f'shape mismatch: {pre_image.shape} vs {gt_image.shape}'
        
        # 生成当前批次的混淆矩阵
        batch_matrix = self._generate_matrix(gt_image, pre_image)
        
        # 检查矩阵形状是否一致
        if self.confusion_matrix.shape != batch_matrix.shape:
            # 扩展当前混淆矩阵以匹配新矩阵的大小
            new_size = max(self.confusion_matrix.shape[0], batch_matrix.shape[0])
            temp_matrix = np.zeros((new_size, new_size))
            
            # 复制原有数据
            temp_matrix[:self.confusion_matrix.shape[0], :self.confusion_matrix.shape[1]] = self.confusion_matrix
            
            # 更新混淆矩阵
            self.confusion_matrix = temp_matrix
            self.num_classes = new_size  # 更新类别数
            
        # 累加混淆矩阵
        self.confusion_matrix += batch_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    eval = Evaluator(num_class=3)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
