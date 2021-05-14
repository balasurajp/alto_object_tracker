from __future__ import absolute_import

from got10k.experiments import *

from siamfcgan import TrackerSiamFC
#from siamfc import TrackerSiamFC



if __name__ == '__main__':
    # setup tracker
    net_path = 'pretrained/SiamGAN/SiamFC_G_22_model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # setup experiments
    experiments = [
        #ExperimentGOT10k('data/GOT-10k', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        #ExperimentOTB('data/OTB', version=2015),
        ExperimentVOT('/media/root1/hd2/NAV/My Research @ IIT Tirupati/Trackers/Oriented_RPN_For_Tracking/Testing Data/votLT2018', version='LT2018'),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('data/Temple-color-128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=False)
        e.report([tracker.name])
