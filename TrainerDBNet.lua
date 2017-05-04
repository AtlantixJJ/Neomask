--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training for ParallelNet
------------------------------------------------------------------------------]]
require 'math'
local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, config)
  print("Trainer...")
  -- training params
  self.config = config
  self.debug = config.debug
  if self.debug then print("Debug Mode!") end
  if self.config.fix == true then print("FIXED!") end
  self.model = model

  self.criterion = nn.SoftMarginCriterion():cuda()
  self.criterion_class = nn.CrossEntropyCriterion():cuda()
  self.lr = config.lr
  self.branch_epoch = 8
  self.ratio = torch.ones(4) ; self.ratio[1] = 0;
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

  -- params and gradparams
  self.pmb,self.gmb = model.maskBranch:getParameters();
  self.psb,self.gsb = model.scoreBranch:getParameters();
  self.pcb,self.gcb = model.classBranch:getParameters();collectgarbage()
  self.pm,self.gm = model.maskNet:getParameters();collectgarbage()
  self.ps,self.gs = model.scoreNet:getParameters();collectgarbage()
  self.pc,self.gc = model.classNet:getParameters();collectgarbage()
  

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()

  self.LM ={LossMeter(),LossMeter(),LossMeter()}

  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  print("LR: %f" % self.optimState.trunk.learningRate)
  print("Ratio :")
  print(self.ratio)
  self.lossmeter:reset()
  for i=1,3 do self.LM[i]:reset() end

  local imt -- display interval
  if self.debug then imt = 2
  else imt = 200 end

  local timer = torch.Timer()
  
  local grad_class = self.gc
  local param_class = self.pc
  local grad_mask = self.gm
  local param_mask = self.pm
  local grad_score = self.gs
  local param_score = self.ps
  if epoch < self.branch_epoch then
    grad_class = self.gcb
    grad_mask = self.gmb
    grad_score = self.gsb
    param_class = self.pcb
    param_mask = self.pmb
    param_score = self.psb
  end

  -- print(self.criterion_class.output,self.gt:sum());
  local fevalclass = function() return self.criterion_class.output, grad_class end
  local fevalmask  = function() return self.criterion.output,   grad_mask end
  local fevalscore = function() return self.criterion.output,   grad_score end



  for n, sample in dataloader:run(self.ratio) do
    -- copy samples to the GPU
    if self.debug and n > 100 then break end
    if sample ~= nil then
      self:copySamples(sample)
      --if self.debug then print(sample) end
      -- forward/backward
      local params, feval, optimState
      if sample.head == 3 then -- classification
        params = param_class
        feval, optimState = fevalclass, self.optimState.trunk
      elseif sample.head == 1 then -- mask
        params = param_mask
        feval, optimState= fevalmask, self.optimState.mask
      elseif sample.head == 2 then -- score
        params = param_score
        feval, optimState = fevalscore, self.optimState.score
      end
      -- print(sample)
      local outputs = self.model:forward(self.inputs,sample.head, true)
      local lossbatch, gradOutputs
      if sample.head == 3 then
        lossbatch = self.criterion_class:forward(outputs, self.labels, true)
        gradOutputs = self.criterion_class:backward(outputs, self.labels)
      else
        lossbatch = self.criterion:forward(outputs, self.labels)
        gradOutputs = self.criterion:backward(outputs, self.labels)
        if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
      end

      if lossbatch < 10 then
        -- updates
        self.model:zeroGradParameters()
        self.model:backward(self.inputs, gradOutputs, sample.head, epoch < self.branch_epoch)
        optim.sgd(feval, params, optimState)
        if sample.head == 1 then 
          self.model.new_act:updateParameters(optimState.learningRate)
        end

        self.LM[sample.head]:add(lossbatch)
        if n % imt == 0 then
          print("Iter %d Loss Statistics : %.3f %.3f %.3f" % {n, self.LM[1]:value(), self.LM[2]:value(), self.LM[3]:value()})
        end
      else
        print("Loss failed. HEAD : %d" % sample.head)
      end
    else
      print("NIL")
    end
  end

  local logepoch
  -- write log
  logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | Mask Loss: %.4f | Score Loss: %.4f | Class Loss: %.4f',
      epoch, timer:time().real/dataloader:size(),self.LM[1]:value(), self.LM[2]:value(), self.LM[3]:value() )
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch), self.modelsv)
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
  self.lossmeter:reset()
  local cnt = 0
  local tot = 0

  for n, sample in dataloader:run() do
    -- copy input and target to the GPU
    if self.debug and n > 100 then break end
    if sample ~= nil then
      self:copySamples(sample)

      if sample.head == 1 then
        local outputs = self.model:forward(self.inputs,sample.head, false)
        self.maskmeter:add(outputs:view(self.labels:size()),self.labels)
      elseif sample.head == 2 then
        local outputs = self.model:forward(self.inputs, sample.head, false)
        self.scoremeter:add(outputs, self.labels)
      elseif sample.head == 3 then
        local outputs = self.model:forward(self.inputs, sample.head, false)
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
      '| score acc %06.2f | class acc %06.2f | bestmodel %s',
      epoch,
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
      self.scoremeter:value(), cnt/tot*100, bestmodel and '*' or 'x')
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
    local ratios = {
      {   1,  20, torch.Tensor({0,1,20,10})},
      {   21, 50, torch.Tensor({0,1,1,1})},
      {   51, 120,torch.Tensor({0,2,1,1})},
      {   121,1e8,torch.Tensor({0,1,1,1})}
    }

    for i, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
        self.ratio = ratios[i][3]
      end
    end
  end
end

return Trainer
