from __future__ import absolute_import
import __init_paths

from got10k.experiments import *

from siamfcgan import TrackerSiamFC


if __name__ == '__main__':
    # setup tracker
    net_path = '/home/ee15b017/Desktop/altov1.1_ED/CODE/models/b32_weighted/SiamFC_G_50_model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # setup experiments
    experiments = [
        #ExperimentGOT10k('/media/hd2/suraj/vot/EvalData/GOT_10K', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        #ExperimentOTB('/media/hd1/suraj/vot/EvalData/OTB100', version=2015),
        ExperimentVOT('/media/hd1/suraj/vot/EvalData/vot2019', version=2019),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('/media/hd1/suraj/vot/EvalData/TEMPLE_128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=False)
        e.report([tracker.name])
