--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Multi-threaded data loader
------------------------------------------------------------------------------]]

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

--------------------------------------------------------------------------------
-- function: create train/val data loaders
function DataLoader.create(config)
  local loaders = {}
  for i, split in ipairs{'train', 'val'} do
    loaders[i] = M.DataLoader(config, split)
  end

  return table.unpack(loaders)
end

--------------------------------------------------------------------------------
-- function: init
function DataLoader:__init(config, split)
  local function main(idx)
    torch.setdefaulttensortype('torch.FloatTensor')
    local seed = config.seed + idx
    torch.manualSeed(seed)

    paths.dofile('DataSamplerNew.lua')
    _G.ds = DataSampler(config, split)
    return _G.ds:size()
  end

  local threads, sizes = Threads(config.nthreads, main)
  self.threads = threads
  self.__size = sizes[1][1]
  self.batch = config.batch
  self.hfreq = config.hfreq
end

--------------------------------------------------------------------------------
-- function: return size of dataset
function DataLoader:size()
  return math.ceil(self.__size / self.batch)
end

--------------------------------------------------------------------------------
-- function: run
function DataLoader:run_class()
  print("Sampling Classification and Score Only!")
  local threads = self.threads
  local size, batch = self.__size, self.batch

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local bsz = math.min(batch, size - idx + 1)
      threads:addjob(
        function(bsz, hfreq)
          local inputs, labels
          if torch.uniform() > 0.5 then head = 2 else head = 3 end
          
          for i = 1, bsz do  
            local input, label = _G.ds:get(head)
            if not inputs then
              local iSz = input:size():totable()
              local mSz = label:size():totable()
              inputs = torch.FloatTensor(bsz, table.unpack(iSz))
              labels = torch.FloatTensor(bsz, table.unpack(mSz))
            end
            inputs[i]:copy(input)
            labels[i]:copy(label)
          end
          collectgarbage()

          return {inputs = inputs, labels = labels, head = head}
        end,
        function(_sample_) sample = _sample_ end,
        bsz, self.hfreq
      )
      idx = idx + batch
    end
  end

  local n = 0
  local function loop()
    enqueue()
    if not threads:hasjob() then return nil end
    threads:dojob()
    if threads:haserror() then threads:synchronize() end
    enqueue()
    n = n + 1
    return n, sample
  end

  return loop
end

local function head_sampling(ratio)
  local head = 3-- head sampling

  local RAND_NUM = (torch.uniform()) * ratio:sum()
  for i=1,3 do
    if ratio[i] <= RAND_NUM and RAND_NUM <= ratio[i+1] then 
      head = i
    end
  end
  return head
end

function DataLoader:run(ratio)
  if ratio == nil then ratio = torch.ones(4); ratio[1] = 0 end
  for i=2,4 do
    ratio[i] = ratio[i] + ratio[i-1]
  end
  local threads = self.threads
  local size, batch = self.__size, self.batch

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local bsz = math.min(batch, size - idx + 1)
      threads:addjob(
        function(bsz, hfreq)
          local inputs, labels
          local head = head_sampling(ratio)
          
          for i = 1, bsz do
            
            local input, label = _G.ds:get(head)
            if not inputs then
              local iSz = input:size():totable()
              local mSz = label:size():totable()
              inputs = torch.FloatTensor(bsz, table.unpack(iSz))
              labels = torch.FloatTensor(bsz, table.unpack(mSz))
            end
            inputs[i]:copy(input)
            labels[i]:copy(label)
          end
          collectgarbage()

          return {inputs = inputs, labels = labels, head = head}
        end,
        function(_sample_) sample = _sample_ end,
        bsz, self.hfreq
      )
      idx = idx + batch
    end
  end

  local n = 0
  local function loop()
    enqueue()
    if not threads:hasjob() then return nil end
    threads:dojob()
    if threads:haserror() then threads:synchronize() end
    enqueue()
    n = n + 1
    return n, sample
  end

  return loop
end

return M.DataLoader
