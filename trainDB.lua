require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')
local utils = paths.dofile('modelUtils.lua')
local default_config = paths.dofile('getconfig.lua')
local resnet

torch.setdefaulttensortype('torch.FloatTensor') 
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)
---- fixing arguments
print("Selecting depth to be", default_config.layer)

---- Select which model to use
local select_model = default_config.modeldef -- "para_model.lua"
local t = select_model:find("/")
if t ~= nil then
  select_model = select_model:sub(t+1)
end
print("Choose model : %s" % {select_model})
paths.dofile(select_model)

if select_model:find("dm") then
  default_config.iSz = 192
  default_config.oSz = 56
  default_config.gSz = 112
end
if select_model:find("para") then
  default_config.iSz = 224
  default_config.oSz = 56
  default_config.gSz = 112
end

default_config.batch = math.floor(default_config.batch * default_config.nGPU * 0.7)
print("BatchSize : %d " % default_config.batch)

----- Activation function of new layer
print("Using activation :")
if default_config.activation == "ReLU" then
  default_config.activation = cudnn.ReLU
elseif default_config.activation == "Sigmoid" then
  default_config.activation = cudnn.Sigmoid
else
  default_config.activation = cudnn.Tanh
end
print(default_config.activation())

---- set GPU
default_config.gpu2 = default_config.gpu
default_config.gpu1 = default_config.gpu
print("Using GPU %d" % default_config.gpu)

local epoch = 1
if #default_config.reload > 0 then
  print("Building from ParallelNet pretrained model...")
  if paths.filep(default_config.reload..'/log') then
    for line in io.lines(default_config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', default_config.reload))
  local m = torch.load(string.format('%s/model.t7', default_config.reload))
  MNet, config = m.model, m.config
  MNet:set_config(default_config)
  if MNet.from_old ~= nil then
    print("From Old")
    MNet:from_old()
  end
  MNet:make_nets()

  --MNet:make_NewActivation(default_config.activation)
  --MNet:set_config(default_config)

  --MNet:change_Activation(default_config.activation)
  --if MNet.from_old ~= nil then MNet:from_old() end

elseif #default_config.trans_model < 3 then
  print("Building from ResNet...")
  resnet = torch.load("pretrained/resnet-50.t7")
  if select_model:find("Deep") ~= nil then
    MNet = nn.DeepMask(default_config)
  else 
    default_config.model = resnet
    MNet = nn.ParallelNet_Mod(default_config,1024)
    default_config.model = None
  end
end

MNet:DataParallel(default_config.nGPU)

---- Loading Trainer
-- paths.dofile('TrainerDBNet.lua')
paths.dofile("TrainerSingle.lua")
local trainer = Trainer(MNet, default_config, 3)

print("Loading Data...")
local DL = paths.dofile('DataLoaderNew.lua')
local TrainDL, ValDL = DL.create(default_config)

epoch = default_config.repoch
--print("| testing")
--trainer:test(epoch,ValDL)
print('| start training from %d ' % {epoch})
for i = 1, 100 do
  
  trainer:train(epoch,TrainDL)
  if i%2 == 0 then trainer:test(epoch,ValDL) end

  epoch = epoch + 1
end
