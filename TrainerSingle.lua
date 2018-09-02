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
paths.dofile("check_points.lua")
paths.dofile("myutils.lua")
local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, config, head)
  print("Single Task Trainer Initializing...")
  -- training params
  self.config = config
  self.debug = config.debug
  if self.debug then print("Debug Mode!") end
  self.model = model
  if self.model.print_model ~= nil then
    self.model:print_model()
  end

  self.criterion = nn.SoftMarginCriterion():cuda()
  self.criterion_class = nn.CrossEntropyCriterion():cuda()
  self.lr = config.lr
  self.act_epoch = 20
  self.branch_epoch = 5
  --self.delay_batch = 8 -- 256
  --print("Delay Batch : %d" % self.delay_batch)
  print("Branch Epoch %d " % self.branch_epoch)
  self.ratio = torch.ones(4) ; self.ratio[1] = 0;
  self:reset_optimState()

  -- params and gradparams
  self.head = head
  if head == 3 then
    print("Choosing Class Net")
    self.params,self.grad_params = model.classNet:getParameters();
    print(model.classNet)
  elseif head == 1 then
    self.params,self.grad_params = model.maskNet:getParameters();collectgarbage()
    print(model.maskNet)
  elseif head == 2 then
    print(model.scoreNet)
    self.params,self.grad_params = model.scoreNet:getParameters();collectgarbage()
  end

  self:summary_parameters(head)

  -- allocate cuda tensors
  -- self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.LM ={LossMeter(),LossMeter(),LossMeter()}
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

  self.inputs = cutorch.createCudaHostTensor()
  self.labels = torch.CudaTensor()

  print("Cloning weight")
  --self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
  print("Trainer init done.")
end

function Trainer:summary_parameters(head)
    if head == 1 then
        print("|MaskNet    :  %d " % self.params:size()[1])
    elseif head == 3 then
        print("|ClassNet : %d " % self.params:size()[1])
    elseif head == 2 then
        print("|ScoreNet : %d " % self.params:size()[1])
    end
end

function Trainer:reset_optimState()
  print("Reset optimState")
  self.optimState = {}
  for k,v in pairs({'optim'}) do
    self.optimState[v] = {
      learningRate = self.config.lr,
      learningRateDecay = 0,
      momentum = self.config.momentum,
      dampening = 0,
      nesterov = true,
      weightDecay = self.config.wd,
    }
  end
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  print("Start training")
  self.model:training()
  cutorch.synchronize()
  self.lossmeter:reset()
  
  self:updateScheduler(epoch)
  print("LR : (%.5f)" % {self.optimState.optim.learningRate})
  print("Ratio :")
  print(self.ratio)
  print("UseAct : true")

  local imt -- display interval
  if self.debug then imt = 2
  else imt = 100 end

  local timer = torch.Timer()
  local bcnt = 0

  local feval = function() return self.criterion.output, self.grad_params end

  for n, sample in dataloader:run(self.ratio) do
    -- copy samples to the GPU
    if self.debug and n > 100 then break end

    if sample ~= nil then
      self:copySamples(sample)
      cutorch.synchronize()
      collectgarbage()
      if self.debug then print(sample) end
      -- useAct = ( torch.uniform() > 0.5 ) and ( epoch > self.act_epoch)
      self.useAct = true

      ----- FORWARD PASS ------
      -- (inputs, head, ifTrain, ifBranch, useAct)
      local outputs = self.model:forward(self.inputs,sample.head, true, epoch < self.branch_epoch, self.useAct)
      if self.debug then print(outputs:size(),outputs:max(),outputs:min()) end
      ----- FORWARD PASS ------
      
      local lossbatch, gradOutputs
      if sample.head == 3 then -- classification : crossentropy
        lossbatch = self.criterion_class:forward(outputs, self.labels)
        gradOutputs = self.criterion_class:backward(outputs, self.labels)
      else
        lossbatch = self.criterion:forward(outputs, self.labels)
        gradOutputs = self.criterion:backward(outputs, self.labels)
        if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
      end
      

      if lossbatch < 10 then -- to avoid exploding gradients      
        if self.debug then print(lossbatch) end
        
        ----- BACKWARD PASS ----
        
        self.model:zeroGradParameters(sample.head)
        if self.debug then print(gradOutputs:size()) end
        -- (inputs, head, ifTrain, ifBranch, useAct)
        r = self.model:backward(self.inputs, gradOutputs, sample.head, false, self.useAct)
        if self.debug then print(r) end
        ----- BACKWARD PASS ----

        ----- UPDATE -----
        -- if epoch < self.branch_epoch then optim.adam(feval, params, optimState)
        -- else optim.sgd(feval, params, optimState) end  
        --if epoch < 5 then optim.adam(feval, params)
        --else optim.sgd(feval, params, optimState) end

        --[[ DELAY BATCH TRAINING
          bcnt = bcnt + 1
          if bcnt == self.delay_batch then
            if epoch < self.branch_epoch then
              optim.sgd(feval, params, optimState)
            else
              optim.sgd(feval, params, optimState)
            end
            self.model:zeroGradParameters()
            bcnt = 0
          end
        ]]

        if self.debug then print("Updating") end
        optim.sgd(feval, self.params, self.optimState.optim)

        if sample.head == 1 and useAct then 
          self.model.new_act:updateParameters(self.optimState.optim.learningRate)
        end
        cutorch.synchronize()
        ----- UPDATE -----

        self.lossmeter:add(lossbatch)
        if n % imt == 0 then
          print("Iter %d Loss Statistics : %.5f" % {n, self.lossmeter:value()})
        end
      else
        print("Loss failed. %f" % lossbatch)
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
  local tosave = {copy_core(self.model, self.model.main_modules),1}
  torch.save(string.format('%s/model.t7', self.rundir),tosave)
  if epoch%50 == 0 then
    --torch.save(string.format('%s/model_%d.t7', self.rundir, epoch), self.modelsv)
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),tosave)
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
      cutorch.synchronize()
      self:copySamples(sample)

      if sample.head == 1 then
        local outputs = self.model:forward(self.inputs,sample.head, false, false, self.useAct)
        self.maskmeter:add(outputs:view(self.labels:size()),self.labels)
      elseif sample.head == 2 then
        local outputs = self.model:forward(self.inputs, sample.head, false, false, self.useAct)
        self.scoremeter:add(outputs, self.labels)
      elseif sample.head == 3 then
        local outputs = self.model:forward(self.inputs, sample.head, false, false, self.useAct)
        local conf, ind = torch.max(outputs,2) -- take maximium along axis 2
        ind = ind:int()
        lbl = self.labels:int()

        cnt = cnt + torch.eq(lbl,ind):sum()
        tot = tot + self.labels:size(1)
      end
      
      if self.debug then print(n) end
    end
  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.maskmeter:value('0.7')
  
  if z > maxacc then
    --torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    local tosave = {copy_core(self.model, self.model.main_modules),1}
    torch.save(string.format('%s/bestmodel.t7', self.rundir),tosave)
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

local function getCudaTensorType(tensorType)
    if tensorType == 'torch.CudaHalfTensor' then
        return cutorch.createCudaHostHalfTensor()
    elseif tensorType == 'torch.CudaDoubleTensor' then
        return cutorch.createCudaHostDoubleTensor()
    else
        return cutorch.createCudaHostTensor()
    end
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
    --local batch_scale = 256 / self.config.batch
    local original = {
      {   1,  30, 1e-2, 5e-4},
      {  31,  50, 1e-3, 5e-4},
      {  51, 60, 5e-4, 5e-4},
      {  61, 80, 1e-4, 5e-4},
      {  81, 120,1e-5, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }
    local ratios = {
      {   1,  20, torch.Tensor({0,5,1,0})},
      {   21, 30, torch.Tensor({0,0,1,10})},
      {   31, 40, torch.Tensor({0,0,1,10})},
      {   41, 50, torch.Tensor({0,5,1,0})},
      {   51, 90, torch.Tensor({0,0,1,10})},
      {   91, 120,torch.Tensor({0,2,1,2})},
      {   121,1e8,torch.Tensor({0,1,1,1})}
    }
    -- FT Parameters
    local ratios = {
      {   1,  50, torch.Tensor({0,0,0,1})},
      {   51, 60, torch.Tensor({1,0,0})},
      {   61, 1e8, torch.Tensor({0,0,0,1})}
    }

    for i, row in ipairs(original) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]-- / self.delay_batch; 
          v.weightDecay=row[4]
        end
        
      end
    end
    for i, row in ipairs(ratios) do
      if epoch >= row[1] and epoch <= row[2] then
        self.ratio = ratios[i][3]
      end
    end

    --self.optimState.class.learningRate = 0.05 * math.pow(0.1, math.floor(epoch / 20))
  end
end

return Trainer
