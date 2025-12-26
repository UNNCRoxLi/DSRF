import os


def init_acc_regime(dataname):
    if 'RAVEN' in dataname:
        return init_acc_regime_raven()
    else:
        return None


def update_acc_regime(dataname, acc_regime, model_output, target, structure_encoded, data_file):
    if 'RAVEN' in dataname:
        update_acc_regime_raven(acc_regime, model_output, target, data_file)
    else:
        return None


def init_acc_regime_raven():
    acc_regime = {"center_single": [0, 0],
                  "distribute_four": [0, 0],
                  "distribute_nine": [0, 0],
                  "in_center_single_out_center_single": [0, 0],
                  "in_distribute_four_out_center_single": [0, 0],
                  "left_center_single_right_center_single": [0, 0],
                  "up_center_single_down_center_single": [0, 0],
                  }
    return acc_regime

def update_acc_regime_raven(acc_regime, model_output, target, data_file):
    acc_one = model_output.data.max(1)[1] == target
    for i in range(model_output.shape[0]):
        regime = data_file[i].split('\\' if os.name == 'nt' else '/')[0]
        acc_regime[regime][0] += acc_one[i].item()
        acc_regime[regime][1] += 1
