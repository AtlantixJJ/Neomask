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

---- fixing arguments
print("Selecting depth to be", default_config.layer)

---- Use small batch to reduce memory cost
if paths.extname("BATCHSMALL_SIGN") ~= nil then
  print("Using small batch...")
  default_config.batch = 16
end

---- Select which model to use
local select_model = default_config.modeldef -- "para_model.lua"
local t = select_model:find("/")
if t ~= nil then
  select_model = select_model:sub(t+1)
end
if select_model:find("dm") then
  default_config.iSz = 192
  default_config.oSz = 56
  default_config.gSz = 112
end
if select_model:find("para") then
  default_config.iSz = 224
  default_config.oSz = 56
  default_config.gSz = 224
end
print("Choose model : %s" % {select_model})
paths.dofile(select_model)

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
cutorch.setDevice(default_config.gpu)

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
  if MNet.precheck_class ~= nil then
    MNet:precheck_class()
    MNet:precheck_mask()
  end
  if MNet.print_model ~= nil then
    MNet:print_model()
  end
  --MNet:change_Activation(default_config.activation)
  --if MNet.from_old ~= nil then MNet:from_old() end

elseif #default_config.trans_model < 3 then
  print("Building from ResNet...")
  resnet = torch.load("pretrained/resnet-50.t7")
  if select_model:find("Deep") ~= nil then
    MNet = nn.DeepMask(default_config)
  else 
    default_config.model = resnet
    MNet = nn.ParallelNet_Mod(default_config)
    default_config.model = None
  end
else
  print("Building from Deepmask : %s" % default_config.trans_model)
  DM, config = torch.load(string.format('%s/model.t7', default_config.trans_model))
  MNet = nn.ParallelNet_Mod()
  MNet:from_DeepMask(DM.model)
  MNet:set_config(default_config)
  if MNet.print_model ~= nil then
    MNet:print_model()
  end
  if MNet.precheck_class ~= nil then
    MNet:precheck_class()
    MNet:precheck_mask()
  end

end

---- Loading Trainer
paths.dofile('TrainerDBNet.lua')
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
