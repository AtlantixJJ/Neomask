--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training and testing loop for SharpMask
------------------------------------------------------------------------------]]

paths.dofile('trainMeters.lua')
require 'cutorch'
require 'optim'
local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  print("Running in %s" % config.rundir)
  print("Optim : %s" % config.optim)
  -- training params
  self.gpu1 = config.gpu1
  self.gpu2 = config.gpu2

  self.model = model
  self.params, self.gradParams = self.model:getParameters()
  self.criterion = criterion
  self.lr = config.lr

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.optimizer = config.optim
  self.optimState = {learningRate=0.01,momentum=0.9,weightDecay=0.01,}

  -- self.optimState

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
  local imt = 100
  local lossum = 0
  local lossbatch = 0
  
  for n, sample in dataloader:run() do
      -- copy samples to the GPU
      cutorch.setDevice(self.gpu1)
      self:copySamples(sample)

      function feval(params)
          self.gradParams:zero()

          local outputs = self.model:forward(self.inputs)
          local label_on2 = self.labels:clone()
          lossbatch = self.criterion:forward(outputs, label_on2)
          if lossbatch < 10 then
              lossum = lossum + lossbatch
              -- print(sample,lossbatch)
              if n % imt == 0 then
                print("Iter %d Loss %.2f " % {n, lossum/imt})
                lossum = 0
              end

              local gradOutputs = self.criterion:backward(outputs, label_on2)
              gradOutputs:mul(self.inputs:size(1))
              self.gradParams:zero()
              -- automatically change to gpu2
              self.model:backward(self.inputs, gradOutputs)
          else
              lossbatch = 0
          end
          return lossbatch, self.gradParams
      end

      if self.optimizer == "adam" then
          optim.adam(feval,self.params)
      elseif self.optimizer == "sgd" then
          optim.sgd(feval,self.params,self.optimState)
      end
      if n % 10 == 0 then
        collectgarbage()
      end
      -- update loss
      self.lossmeter:add(lossbatch)
  end

  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
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


  for n, sample in dataloader:run() do
    -- copy input and target to the GPU
    cutorch.setDevice(self.gpu1)
    self:copySamples(sample)

    -- infer mask in batch
    local outputs = self.model:forward(self.inputs):float()
    cutorch.synchronize()

    self.maskmeter:add(outputs:view(sample.labels:size()),sample.labels)

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
      '| bestmodel %s',
      epoch,
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
      bestmodel and '*' or 'x')
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
      {  1,     9,  5e-3},
      {  10,    20,  1e-3},
      { 21,     50,  5e-4},
      { 51,    1e8,  1e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        self.lr = row[3]
      end
    end
  end
end

return Trainer
