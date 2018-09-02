--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

When initialized, it creates/load the common trunk, the maskBranch and the
scoreBranch.
DeepMask class members:
  - trunk: the common trunk (modified pre-trained resnet50)
  - maskBranch: the mask head architecture
  - scoreBranch: the score head architecture
------------------------------------------------------------------------------]]

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
if nn.SpatialSymmetricPadding == nil then
    paths.dofile('SpatialSymmetricPadding.lua')
end
local utils = paths.dofile('modelUtils.lua')

local DeepMask,_ = torch.class('nn.DeepMask','nn.Container')

--------------------------------------------------------------------------------
-- function: constructor
function DeepMask:__init(config)
  -- create common trunk
  self:createTrunk(config)

  -- create mask head
  self:createMaskBranch(config)

  -- create score head
  self:createScoreBranch(config)

  self.maskNet = nn.Sequential():add(self.trunk):add(self.maskBranch)
  self.scoreNet = nn.Sequential():add(self.trunk):add(self.scoreBranch)

  -- number of parameters
  local npt,nps,npm = 0,0,0
  local p1,p2,p3  = self.trunk:parameters(),
    self.maskBranch:parameters(),self.scoreBranch:parameters()
  for k,v in pairs(p1) do npt = npt+v:nElement() end
  for k,v in pairs(p2) do npm = npm+v:nElement() end
  for k,v in pairs(p3) do nps = nps+v:nElement() end
  print(string.format('| number of paramaters trunk: %d', npt))
  print(string.format('| number of paramaters mask branch: %d', npm))
  print(string.format('| number of paramaters score branch: %d', nps))
  print(string.format('| number of paramaters total: %d', npt+nps+npm))
end

function DeepMask:forward(input_batch, head)
  if head == 1 then
    self.output = self.maskNet:forward(input_batch)
  else
    self.output = self.scoreNet:forward(input_batch)
  end
  return self.output
end

function DeepMask:backward(input_batch, gradOutput, head)
  local grad
  if head == 1 then
    grad = self.maskNet:backward(input_batch, gradOutput)
  else
    grad = self.scoreNet:backward(input_batch, gradOutput)
  end
  return grad
end

--------------------------------------------------------------------------------
-- function: create common trunk
function DeepMask:createTrunk(config)
  -- size of feature maps at end of trunk
  self.fSz = config.iSz/16
  print("Input Size : ", config.iSz)
  print("Raw output size : ", config.oSz)
  print("Ground Truth Size : ", config.gSz)

  -- load trunk
  local trunk = torch.load('pretrained/resnet-50.t7')

  -- remove BN
  utils.BNtoFixed(trunk, true)

  -- remove fully connected layers
  trunk:remove();trunk:remove();trunk:remove();trunk:remove()
  

  -- crop central pad
  -- MODIFIED
  trunk:add(nn.SpatialZeroPadding(-1,-1,-1,-1))
  local tempin = torch.rand(1,3,config.iSz,config.iSz)
  local res = trunk:cuda():forward(tempin:cuda())
  print(res:size())
  -- add common extra layers
  trunk:add(cudnn.SpatialConvolution(1024,128,1,1,1,1))
  trunk:add(cudnn.ReLU(true))
  trunk:add(nn.View(config.batch,128*self.fSz*self.fSz))
  trunk:add(nn.Linear(128*self.fSz*self.fSz,512))

  -- from scratch? reset the parameters
  if config.scratch then
    for k,m in pairs(trunk.modules) do if m.weight then m:reset() end end
  end

  -- symmetricPadding
  utils.updatePadding(trunk, nn.SpatialSymmetricPadding)

  self.trunk = trunk:cuda()
  return trunk
end

--------------------------------------------------------------------------------
-- function: create mask branch
function DeepMask:createMaskBranch(config)
  local maskBranch = nn.Sequential()

  -- maskBranch
  maskBranch:add(nn.Linear(512,config.oSz*config.oSz))
  self.maskBranch = nn.Sequential():add(maskBranch:cuda())

  -- upsampling layer
  if config.gSz > config.oSz then
    local upSample = nn.Sequential()
    upSample:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
    upSample:add(nn.View(config.batch,config.oSz,config.oSz))
    upSample:add(nn.SpatialReSamplingEx{owidth=config.gSz,oheight=config.gSz,
    mode='bilinear'})
    upSample:add(nn.View(config.batch,config.gSz*config.gSz))
    upSample:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
    self.maskBranch:add(upSample)
  end

  return self.maskBranch
end

--------------------------------------------------------------------------------
-- function: create score branch
function DeepMask:createScoreBranch(config)
  local scoreBranch = nn.Sequential()
  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(512,1024))
  --scoreBranch:add(nn.Threshold(0, 1e-6))
  scoreBranch:add(cudnn.ReLU(true))
  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(1024,1))

  self.scoreBranch = scoreBranch:cuda()
  return self.scoreBranch
end

--------------------------------------------------------------------------------
-- function: training
function DeepMask:training()
  self.trunk:training(); self.maskBranch:training(); self.scoreBranch:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function DeepMask:evaluate()
  self.trunk:evaluate(); self.maskBranch:evaluate(); self.scoreBranch:evaluate()
end

--------------------------------------------------------------------------------
-- function: to cuda
function DeepMask:cuda()
  self.trunk:cuda(); self.scoreBranch:cuda(); self.maskBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function DeepMask:float()
  self.trunk:float(); self.scoreBranch:float(); self.maskBranch:float()
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function DeepMask:inference()
  self.trunk:evaluate()
  self.maskBranch:evaluate()
  self.scoreBranch:evaluate()

  utils.linear2convTrunk(self.trunk,self.fSz)
  utils.linear2convHead(self.scoreBranch)
  utils.linear2convHead(self.maskBranch.modules[1])
  self.maskBranch = self.maskBranch.modules[1]

  self:cuda()
end

--------------------------------------------------------------------------------
-- function: clone
function DeepMask:clone(...)
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self); f:seek(1)
  local clone = f:readObject(); f:close()

  if select('#',...) > 0 then
    clone.trunk:share(self.trunk,...)
    clone.maskBranch:share(self.maskBranch,...)
    clone.scoreBranch:share(self.scoreBranch,...)
  end

  return clone
end

return nn.DeepMask
