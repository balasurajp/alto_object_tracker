
class Config:
    def __init__(self):
        # Baseline pretrained model
        self.pretrained_model_path = './models/siampretrain/pretrained.pth'
        
        # Logistics Info
        self.use_gpu = True
        self.gpu_id = 1
        self.log_freq = 70
        self.resume = False
        self.start_epoch = 0
        if not self.resume:
            assert self.start_epoch == 0

        # Datapath
        self.save_base_path = './snapshots'
        self.save_sub_path = 'alto_v1'
        self.INV_DATA = "./ILSVRC2015_curation/Data/VID/train"
        self.INV_IMDB = "./ILSVRC15-curation/imdb_video_train.json"
        self.GOT10K_DATA = "./ILSVRC2015_curation/Data/VID/train"
        self.GOT10K_IMDB = "./ILSVRC15-curation/imdb_video_train.json"

        # =====================parameters for training========================
        # augmentation
        self.DATASET.SHIFT = 4
        self.DATASET.SCALE = 0.05
        self.DATASET.COLOR = 1
        self.DATASET.FLIP = 0
        self.DATASET.BLUR = 0
        self.DATASET.ROTATION = 0

        self.TEMPLATE_SIZE = 127
        self.SEARCH_SIZE = 255
        self.STRIDE = 8
        self.rPos = 16
        self.rNeg = 0
        self.label_weight_method = "balanced"


        self.PAIR_RANGE = 100
        self.TRAIN_PAIRS = 5.32e4
        self.num_epoch = 50
        self.batch_size = 40
        
        self.sub_mean = 0
        self.train_num_workers = 8  # number of threads to load data
        self.val_num_workers = 8


        self.lr = 1                  # learning rate of SGD
        self.momentum = 0.9          # momentum of SGD
        self.weight_decay = 0.0001   #self.weight_decay = 5e-4     # weight decay of optimizator
        self.bn_momentum = .0003
       
        # Optimization function Info 
        self.loss = 'logistic'
        self.neg_label = 0

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
        self.bbox_output_path = "./tracking_result/"

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 17

        # saving path for snapshots in training
        self.net_base_path = "./snapshots"
        # OTB database path
        self.seq_base_path = "../OTB"
        # Pretrained model path
        self.net = "SiamFC_G_8_model.pth"
