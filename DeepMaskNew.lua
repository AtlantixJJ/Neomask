--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

local utils = paths.dofile('modelUtils.lua')
paths.dofile("para_model.lua")
paths.dofile("DeepMask.lua")
local ParallelNet_DM,_ = torch.class("nn.ParallelNet_DM",'nn.ParallelNet')

function ParallelNet_DM:set_config(config)
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2
    assert(self.gpu1==self.gpu2)
    cutorch.setDevice(self.gpu1)
    self.traceback_depth = config.layer - 1

    self.config = config

    self.name = config.name
    self.softmax = nn.SoftMax():cuda()
end

function ParallelNet_DM:__init(config,NFC)
    if config == nil then
        print("NIL Initialization. Return.")
        return
    end
    self:set_config(config)
    self.tSz = 56
    self.NFC = NFC
    print("Using NFC",self.NFC)

    -- the preset model cannot be cuda model
    self.M = config.model
    utils.BNtoFixed(self.M, true)
    self.M:remove();self.M:remove();self.M:remove();self.M:remove();
    self.M:add(nn.SpatialZeroPadding(-1,-1,-1,-1))
    self.M = self.M:cuda()
    self.fSz = self.M:forward(torch.rand(1, 3, self.config.iSz, self.config.iSz):cuda()):size(3)
    -- if 4-remove, then start size is 1024,14,14
    -- 1024,8,8

    self:build_trunk()
    self:build_classBranch()
    self:build_scoreBranch()

    -- pre-running to determine shapes
    self:precheck_class()
    self:build_maskBranch()
    self:precheck_mask()

    self.classNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.classBranch)
    self.scoreNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.scoreBranch)
    self.maskNet  = nn.Sequential():add(self.trunk):add(self.maskBranch)
    collectgarbage()

    self:make_NewActivation(self.config.activation) 
end

function ParallelNet_DM:build_trunk()
    print("ParallelNet_DM build trunk...")
    self.trunk = nn.Sequential():add(self.M)

    self.trunk:add(cudnn.SpatialConvolution(1024,128,1,1,1,1))
    self.trunk:add(cudnn.ReLU(true))
    self.trunk:add(nn.View(-1,128*self.fSz*self.fSz))
    self.trunk:add(nn.Linear(128*self.fSz*self.fSz,self.NFC))
    self.trunk:add(cudnn.ReLU(true))
    self.trunk = self.trunk:cuda()

    self.trunk_head = nn.Sequential()
    self.trunk_head:add(nn.Dropout(.5))
    self.trunk_head:add(nn.Linear(self.NFC,self.NFC))
    self.trunk_head:add(nn.Threshold(0, 1e-6))
    self.trunk_head = self.trunk_head:cuda()

    self.trunkNet = nn.Sequential():add(self.trunk):add(self.trunk_head)
end

function ParallelNet_DM:build_scoreBranch()
    self.scoreBranch = nn.Sequential():add(nn.Linear(self.NFC,1)):cuda()
end

function ParallelNet_DM:build_classBranch()
    self.classBranch = nn.Sequential():add(nn.Linear(self.NFC,90)):cuda()
    --- For convenience of backward propagation
    self.trunk_head_class = nn.Sequential():add(self.trunk_head):add(self.classBranch)
end

function ParallelNet_DM:build_maskBranch()
    print("ParallelNet_DM build mask...")
    print("Raw output size : ", self.config.oSz)
    print("Ground Truth Size : ", self.config.gSz)
    self.maskBranch = nn.Sequential()

    -- maskBranch
    self.maskBranch:add(nn.Linear(self.NFC, self.config.oSz * self.config.oSz))
    self.maskBranch:add(nn.View(-1, self.config.oSz, self.config.oSz))
    self.maskBranch:add(nn.SpatialUpSamplingBilinear(self.scale))
    self.maskBranch:add(nn.View(-1,self.config.gSz * self.config.gSz))
    self.maskBranch = self.maskBranch:cuda()
end

return nn.ParallelNet_Mod