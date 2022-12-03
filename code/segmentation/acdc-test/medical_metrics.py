from medpy import metric
import numpy as np


class Medical_Metric():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.class_dice = 0.0
        self.class_hdp5 = 0.0
        # self.mean_dice = 0.0
        self.n = 0

    def update(self, label_preds, label_trues):
        """
        2d的
        :param label_trues:
        :param label_preds:
        :return:
        """
        #  记录每个类的dice,
        for label_pred, label_true in zip(label_preds, label_trues):
            #  记录每个类的dice,
            class_dice = []
            class_hdp5 = []
            for i in range(1, self.n_classes):
                result = self.calculate_metric_percase(label_pred == i, label_true == i)
                class_dice.append(result[0])
                class_hdp5.append(result[1])
            self.class_dice += np.array(class_dice)
            self.class_hdp5 += np.array(class_hdp5)
            # self.mean_dice += sum(class_dice) / len(class_dice)
            self.n = self.n + 1

    def get_results(self):
        class_hdp5 = self.class_hdp5 / self.n
        class_dice = self.class_dice / self.n
        mean_dice1 = np.mean(class_dice)
        mean_hd952 = np.mean(class_hdp5)

        print("n:", self.n)
        print("mean_dice : {:.5f}".format(mean_dice1))
        print("hd952 : {:.5f}".format(mean_hd952))
        for i, item in enumerate(class_dice):
            print("class {}: {:.5f}".format(i, item))
        return {
            "mean_dice": mean_dice1,
            "mean_hd952": mean_hd952,
            "class_dice": class_dice,
        }

    def reset(self):
        self.class_dice = 0.0
        self.class_hdp5 = 0.0
        # self.mean_dice = 0.0
        self.n = 0

    def to_str(self, metrics):
        print("mean_dice: {:.5f}\t mean_hd952: {:.5f}".format(metrics["mean_dice"], metrics["mean_hd952"]))
        for i, item in enumerate(metrics["class_dice"]):
            print("class:{} \t dice{:.5f}".format(i + 1, item))

    @staticmethod
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95
        elif pred.sum() > 0 and gt.sum() == 0:
            return 1, 0
        else:
            return 0, 0
