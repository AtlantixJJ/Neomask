--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training for simulated Deepmask
------------------------------------------------------------------------------]]
require 'math'
local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  -- training params
  self.config = config
  if self.config.fix == true then print("FIXED!") end
  self.model = model

  self.criterion = criterion
  self.criterion_class = nn.CrossEntropyCriterion():cuda()
  self.lr = config.lr
  self.optimState ={}
  for k,v in pairs({'trunk','mask','score'}) do
    self.optimState[v] = {
      learningRate = config.lr,
      learningRateDecay = 0,
      momentum = config.momentum,
      dampening = 0,
      weightDecay = config.wd,
    }
  end
  self.optimState.trunk.learningRate = 0.1

  -- params and gradparams
  self.pm,self.gm = model.maskBranch:getParameters();collectgarbage()
  self.ps,self.gs = model.scoreBranch:getParameters();collectgarbage()
  self.pt,self.gt = model.M:getParameters();collectgarbage()
  

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

  -- log
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()

  local timer = torch.Timer()
  -- print(self.criterion_class.output,self.gt:sum());
  local fevalclass = function()  return self.criterion_class.output,   self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  local fevalscore = function() return self.criterion.output,   self.gs end

  if epoch > 100 then
    for n, sample in dataloader:run() do
      -- copy samples to the GPU
      self:copySamples(sample)

      -- forward/backward
      local params, feval, optimState
      if sample.head == 3 then
        params = self.pt
        feval, optimState = fevalclass, self.optimState.trunk
      elseif sample.head == 1 then
        params = self.pm
        feval,optimState= fevalmask, self.optimState.mask
      elseif sample.head == 2 then
        params = self.ps
        feval,optimState = fevalscore, self.optimState.score
      end

      local outputs = self.model:forward(self.inputs,sample.head)
      local lossbatch, gradOutputs
      if sample.head == 3 then lossbatch = self.criterion_class:forward(outputs, self.labels)
      else lossbatch = self.criterion:forward(outputs, self.labels) end

      if lossbatch < 10 then
        self.model:zeroGradParameters()

        if sample.head == 3 then gradOutputs = self.criterion_class:backward(outputs, self.labels)
        else gradOutputs = self.criterion:backward(outputs, self.labels) end

        if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
        self.model:backward(self.inputs, gradOutputs, sample.head)

        -- optim.sgd(feval, params, optimState)
        self.model:updateParameters(optimState.learningRate)

        -- update loss
        self.lossmeter:add(lossbatch)
        print("HEAD %d Loss %.3f" % {sample.head, lossbatch})
      else
        print("Loss failed. HEAD : %d" % sample.head)
      end
    end
  else
    print("LR:%f"%self.optimState.trunk.learningRate)
    for n, sample in dataloader:run_class_only() do
      -- copy samples to the GPU
      self:copySamples(sample)

      -- forward/backward
      local params, feval, optimState
      if sample.head == 3 then
        params = self.pt
        feval, optimState = fevalclass, self.optimState.trunk
      elseif sample.head == 1 then
        params = self.pm
        feval,optimState= fevalmask, self.optimState.mask
      elseif sample.head == 2 then
        params = self.ps
        feval,optimState = fevalscore, self.optimState.score
      end

      local outputs = self.model:forward(self.inputs,sample.head)
      local lossbatch, gradOutputs
      if sample.head == 3 then lossbatch = self.criterion_class:forward(outputs, self.labels)
      else lossbatch = self.criterion:forward(outputs, self.labels) end

      if lossbatch < 10 then
        self.model:zeroGradParameters()

        if sample.head == 3 then gradOutputs = self.criterion_class:backward(outputs, self.labels)
        else gradOutputs = self.criterion:backward(outputs, self.labels) end

        if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
        self.model:backward(self.inputs, gradOutputs, sample.head)

        optim.sgd(feval, params, optimState)
        -- self.model:updateParameters(optimState.learningRate)

        -- update loss
        self.lossmeter:add(lossbatch)
        print("HEAD %d Loss %.3f" % {sample.head, lossbatch})
      else
        print("Loss failed. HEAD : %d" % sample.head)
      end
    end
  end
  local logepoch
  
  -- write log
  if self.config.fix == false then
    logepoch =
      string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
        epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  else
    logepoch =
      string.format('[FIXED train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
        epoch, timer:time().real/dataloader:size(),self.lossmeter:value())  
  end
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%10 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
      self.modelsv)
  end

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
local maxacc = 0
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  self.maskmeter:reset()
  self.scoremeter:reset()

  for n, sample in dataloader:run() do
    -- copy input and target to the GPU
    self:copySamples(sample)

    if sample.head == 1 then
      local outputs = self.model:forward(self.inputs,sample.head)
      self.maskmeter:add(outputs:view(self.labels:size()),self.labels)
    else
      local outputs = self.scoreNet:forward(self.inputs, sample.head)
      self.scoremeter:add(outputs, self.labels)
    end
    cutorch.synchronize()

  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.maskmeter:value('0.7')
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f suc@.7 %06.2f '..
      '| acc %06.2f | bestmodel %s',
      epoch,
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
      self.scoremeter:value(), bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 0.1, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]*math.pow(0.5,math.floor((epoch - 1) / 4)); v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
