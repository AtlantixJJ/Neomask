--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training the classification functionality.
Train the new layers in the first 10 epochs, then train the whole network
------------------------------------------------------------------------------]]
require 'math'
local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, config)
  -- training params
  self.config = config
  self.debug = config.debug
  if self.debug then print("Debug Mode!") end
  if self.config.fix == true then print("FIXED!") end
  self.model = model

  self.criterion = nn.SoftMarginCriterion():cuda()
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

  -- parameter / gradient
  self.pt,self.gt = self.model:getParameters(3);collectgarbage()
  self.ps,self.gs = self.model:getParameters(2);collectgarbage()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()
  self.LM ={LossMeter(),LossMeter(),LossMeter()}

  -- log
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  print("Classification Only Training...")
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()
  print("LR:%f"%self.optimState.trunk.learningRate)

  local imt
  if self.debug then imt = 2
  else imt = 100 end

  local timer = torch.Timer()
  -- print(self.criterion_class.output,self.gt:sum());
  if epoch > 20 then 
    print("Using Full Parameters")
    self.pt,self.gt = self.model:getParameters(1)
  end

  local fevalclass = function() return self.criterion_class.output,   self.gt end
  local fevalscore = function() return self.criterion.output,         self.gs end
  local feval

  local lossum = 0
  for n, sample in dataloader:run_class() do
    if self.debug and n > 100 then break end
    if sample ~= nil then
      -- copy samples to the GPU
      self:copySamples(sample)
      -- forward/backward

      local outputs = self.model:forward(self.inputs,sample.head)
      local lossbatch, gradOutputs

      if sample.head == 2 then --score
        lossbatch = self.criterion:forward(outputs, self.labels)
        gradOutputs = self.criterion:backward(outputs, self.labels)
        feval = fevalclass
      elseif sample.head == 3 then --class
        lossbatch = self.criterion_class:forward(outputs, self.labels)
        gradOutputs = self.criterion_class:backward(outputs, self.labels)
        feval = fevalscore
      end

      if lossbatch < 10 then
        lossum = lossum + lossbatch
        self.model:zeroGradParameters()
        self.model:backward(self.inputs, gradOutputs)
        optim.sgd(feval, self.pt, self.optimState.trunk)
        -- update loss
        self.LM[sample.head]:add(lossbatch)
        if self.debug then print(lossbatch) end
        if n % 100 == 0 then 
          print("Iter %d\tLoss(Score) %.3f\tLoss(Acc) %.3f" % {n, self.LM[2]:value(), self.LM[3]:value()})
          lossum = 0
        end
      else
        print("Loss failed. HEAD : %d" % sample.head)
      end
    end
  end

  local logepoch
  
  logepoch =
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
  print("testing")
  self.model:evaluate()
  self.maxacc = self.maxacc or 0
  self.maskmeter:reset()
  self.scoremeter:reset()
  self.lossmeter:reset()
  local cnt = 0
  local tot = 0

  for n, sample in dataloader:run_class() do
    -- copy input and target to the GPU
    if self.debug and n > 100 then break end
    if sample ~= nil then
      self:copySamples(sample)
      if self.debug then print(sample) end

      if sample.head == 2 then
        local outputs = self.model:forward(self.inputs, sample.head)
        self.scoremeter:add(outputs, self.labels)
      elseif sample.head == 3 then
        local outputs = self.model:forward(self.inputs, sample.head)
        local conf, ind = torch.max(outputs,2) -- take maximium along axis 2
        ind = ind:int()
        lbl = self.labels:int()

        cnt = cnt + torch.eq(lbl,ind):sum()
        tot = tot + self.labels:size(1)
      end
    end
  end

  self.model:training()

  -- check if bestmodel so far
  local z = cnt/tot
  local bestmodel = false
  if z > self.maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    self.maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| acc %06.2f | bestmodel %s',
      epoch, z, bestmodel and '*' or 'x')
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
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
