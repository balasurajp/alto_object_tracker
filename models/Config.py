
class Config:
    def __init__(self):
        # LOGISTICS INFORMATION
        self.use_gpu = True
        self.gpuid = 1
        self.logfreq = 28
        self.resume = False
        self.start_epoch = 0
        if not self.resume:
            assert self.start_epoch == 0

        # PATH VARIABLES
        self.cwd = None
        self.snappath = self.cwd + "/snapshots/altov1/"
        self.datapath = self.cwd + "/ILSVRC2015/curation_data/Data/VID/train"
        self.trnimdb = self.cwd + "/ILSVRC2015/curation_jsons/train.json"
        self.valimdb = self.cwd + "/ILSVRC2015/curation_jsons/val.json"
        self.pretrainedpath = self.cwd + "/models/siampretrain/pretrained.pth"

        # =====================TRAINING PARAMETERS========================
        self.template_size = 127
        self.instance_size = 255
        self.stride = 8

        self.batchsize = 32
        self.num_epoch = 40
        self.pairband = 100
        self.numpairs = 5.32e4
        
        self.trn_numworkers = 8  # number of threads to load data
        self.val_numworkers = 8

        # Optimizer configuration
        self.lr = 1                  
        self.betas = (0.9, 0.999)
        self.momentum = 0.9         
        self.weightdecay = 0.0001 #5e-4      # weight decay (0.0001)
        self.bn_momentum = .0003
       
        # Optimization configuration
        self.loss = 'logistic'
        self.rPos = 16
        self.rNeg = 0
        self.neg_label = 0
        self.label_weight_method = "balanced"

        # GAN parameters
        self.initG = 'truncated'
        self.initD = 'truncated'
        self.real_label = 1
        self.fake_label = 0

        # ===================parameters for tracking (SiamFC-3s by default)==========================
        self.num_scale = 3
        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_LR = 0.59
        self.response_UP = 16
        self.windowing = "cosine"
        self.w_influence = 0.176

        self.video = "Lemming"
        self.visualization = 0
        self.bbox_output = True
        self.bbox_outputpath = "./tracking_result/"

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 17

        # saving path for snapshots in training
        self.net_basepath = "./snapshots"
        # OTB database path
        self.seq_basepath = "../OTB"
        # Pretrained model path
        self.net = "SiamFC_G_8_model.pth"
