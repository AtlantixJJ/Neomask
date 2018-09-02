require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

paths.dofile("DeepMaskNew.lua")

local ParallelNet_Mod,parent = torch.class("nn.ParallelNet_Mod",'nn.ParallelNet_DM')

function ParallelNet_Mod:__init(config, NFC)
    print("Parallel Model (DeepMask 1-Linear) Initialization.")
    if config ~= nil then
        config.iSz = 192
        config.gSz = 112
        config.oSz = 56
        print("Fixed Size : ",config.iSz, config.oSz, config.gSz)
    end
    self.fSz = 10

    parent.__init(self, config, NFC)
end

function ParallelNet_Mod:from_DeepMask(DModel)
    --print(DModel)
    self.NFC = 512
    self.trunk = DModel.trunk
    self.trunk.modules[#self.trunk - 1] = nn.View(-1, 128 * 10 * 10):cuda()
    -- the last layer is 128*10*10 => 56*56
    self.maskBranch = DModel.maskBranch
    local branch = self.maskBranch.modules[2]
    self.maskBranch.modules[2] = nn.Sequential()
    self.maskBranch.modules[2]:add(nn.View(-1, 56,56))
    self.maskBranch.modules[2]:add(nn.SpatialUpSamplingBilinear(2))
    self.maskBranch.modules[2]:add(nn.View(-1, 112*112))
    self.maskBranch.modules[2] = self.maskBranch.modules[2]:cuda()
    self.scoreBranch = DModel.scoreBranch:clone()

    self.trunk_head = DModel.scoreBranch:clone()
    self.trunk_head:remove();self.trunk_head:remove();
    for i=1,3 do table.remove(self.scoreBranch.modules,1) end

    -- common layer is (512,)

    self.classBranch = nn.Sequential():add(nn.Dropout(.5)):add(nn.Linear(1024,90)):cuda()
    self.trunk_head_class = nn.Sequential():add(self.trunk_head):add(self.classBranch)
    self:make_NewActivation()

    self.classNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.classBranch)
    self.scoreNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.scoreBranch)
    self.trunkNet = nn.Sequential():add(self.trunk):add(self.trunk_head)
    self.maskNet  = nn.Sequential():add(self.trunk):add(self.maskBranch)
end

function ParallelNet_Mod:make_NewActivation(activation)
    print("ParallelNet make activation (1-Linear)")
    local input_layer = nn.ParallelTable()
    local branch = nn.Sequential()
    activation = activation or cudnn.Sigmoid
    print("Using activation : ")
    print(activation())
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_nfeat, self.from_size, self.from_size) -- use multiple
    --branch:add(cudnn.SpatialFullConvolution(self.from_nfeat,self.from_nfeat,1,1,1,1,0,0,0,0))
    branch:add(nn.Linear(self.NFC,self.NFC))
    branch:add(activation())
    input_layer:add(branch) -- use single scale factor
    --input_layer:add(nn.Sequential():add(nn.CMul(self.from_nfeat, self.from_size, self.from_size)):add(nn.CAdd(self.from_nfeat, self.from_size, self.from_size)):add(activation))
    self.new_act = nn.Sequential():add(input_layer):add(nn.CMulTable()):cuda()
end

return nn.ParallelNet_Mod