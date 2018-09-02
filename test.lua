require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')

local default_config = paths.dofile('getconfig.lua')

-- fixing arguments
print("Selecting depth to be 2")
default_config.layer = 2

if paths.extname("BATCHSMALL_SIGN") ~= nil then
  print("Using small batch...")
  default_config.batch = 16
end

local select_model = default_config.modeldef -- "para_model.lua"
local t = select_model:find("/")
if t ~= nil then
  select_model = select_model:sub(t+1)
end
print("Choose model : %s" % {select_model})
paths.dofile(select_model)

print("Using activation : Sigmoid")
default_config.activation = cudnn.Sigmoid
local utils = paths.dofile('modelUtils.lua')

-- set GPU
default_config.gpu2 = default_config.gpu
default_config.gpu1 = default_config.gpu
print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)

local epoch = 1, denet
if #default_config.reload > 0 then
  print("Building from ParallelNet pretrained model...")
  epoch = 1
  if paths.filep(default_config.reload..'/log') then
    for line in io.lines(default_config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', default_config.reload))
  local m = torch.load(string.format('%s/model.t7', default_config.reload))
  MNet, config = m.model, m.config
  MNet:change_Activation(default_config.activation)

elseif #default_config.trans_model < 3 then
  print("Building from ResNet...")
  local resnet = torch.load("pretrained/resnet-50.t7")

  default_config.model = resnet
  
  MNet = nn.ParallelNet_Mod(default_config,default_config.layer)
  default_config.model = None

else
  print("Building from transfer learned model...")
  if paths.filep(default_config.trans_model..'/log') then
    for line in io.lines(default_config.trans_model..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', default_config.trans_model))
  local m = torch.load(string.format('%s/model.t7', default_config.trans_model))  
  default_config.model = m.model.M
  MNet = nn.ParallelNet_Mod(default_config,default_config.layer)
  default_config.model = None
end

-- Loading Trainer
paths.dofile('TrainerDBNet.lua')
cutorch.setDevice(default_config.gpu2)
local trainer = Trainer(MNet, default_config)

print("Loading Data...")
local DL = paths.dofile('DataLoaderNew.lua')
local TrainDL, ValDL = DL.create(default_config)

epoch = default_config.repoch
print('| start training from %d ' % {epoch})
for i = 1, 100 do
  
  trainer:train(epoch,TrainDL)
  if i%2 == 0 then trainer:test(epoch,ValDL) end

  epoch = epoch + 1
end
