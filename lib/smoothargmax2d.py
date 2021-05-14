class smoothargmax2D(torch.nn.Module):
    def __init__(self):
        super(smoothargmax2D, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        #Beta parameter controls impulse nature of softmax distribution
        self.beta = 5

    def forward(self, x):
    	batchsz, h, w = x.size()

    	# Projecting onto x,y axes by summing over other dimension
    	x_w = x.sum(dim=1)
    	x_h = x.sum(dim=2)

    	# Impulse like softmax distribution
        probx_w = self.softmax(self.beta * x_w)
        probx_h = self.softmax(self.beta * x_h)
        
        # Indices along x,y axes
        idx_w = torch.arange(w, dtype=x.dtype).repeat(batchsz,1)
        idx_h = torch.arange(h, dtype=x.dtype).repeat(batchsz,1)

        # Compute smooth armax along x, y axes
        maxW = (probx_w*idx_w).sum(dim=1).view(batchsz,1)
        maxH = (probh_w*idx_h).sum(dim=1).view(batchsz,1)

        return torch.concat([maxH, maxW], dim=1)