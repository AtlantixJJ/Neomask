--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'

paths.dofile('para_model.lua')

local ParallelNet_Mod,ParallelNet = torch.class("nn.ParallelNet_Mod",'nn.ParallelNet')
function ParallelNet_Mod:__init(config, layer)
    print("Parallel Model (Backward Only Naive) Initialization.")
    ParallelNet.__init(self, config, layer)
end

function ParallelNet_Mod:from_old()
    self:build_maskBranch()
end

function ParallelNet_Mod:build_maskBranch()
    print("Backward MaskBranch")
    self.maskBranch = nn.Sequential()
    local nfeat = 1024 --self.mask_nfeat
    local dt = 0 -- 2
    -- (1024,14,14)
    self.maskBranch:add(cudnn.SpatialConvolution(nfeat, nfeat/4, 3, 3, 1, 1))
    self.maskBranch:add(cudnn.SpatialBatchNormalization(nfeat/4))
    self.maskBranch:add(cudnn.ReLU(true)) -- (256,12,12)
    self.maskBranch:add(cudnn.SpatialConvolution(nfeat/4, nfeat/16, 3, 3, 1, 1))
    self.maskBranch:add(cudnn.SpatialBatchNormalization(nfeat/16))
    self.maskBranch:add(cudnn.ReLU(true)) -- (64,10,10)    
    self.maskBranch:add(nn.View(-1,nfeat/16*10*10))
    self.maskBranch:add(nn.Linear(nfeat/16*10*10, 512))
    self.maskBranch:add(nn.Linear(512,56*56))
    self.maskBranch:add(nn.View(-1,56,56))
    -- the mask prediction use marginal loss, there should not be activation

    self.maskBranch:add(nn.SpatialUpSamplingBilinear( math.floor(self.config.gSz / 56) ))
    self.maskBranch:add(nn.View(-1,self.config.gSz*self.config.gSz))

    self.maskBranch = self.maskBranch:cuda()
end

function ParallelNet_Mod:make_NewActivation()
    print("ParallelNet_Comb Activation Building...")
    self.new_act = nn.Sequential()
    local input_layer = nn.ParallelTable()
    local net_input = nn.Sequential()
    local net_rele = nn.Sequential()

    net_input:add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_rele :add(nn.SpatialSymmetricPadding(1,1,1,1))
    net_input:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))
    net_rele :add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat/2,3,3,1,1))

    input_layer:add(net_input):add(net_rele)
    
    self.new_act:add(input_layer):add(nn.JoinTable(2)) -- join at (batch,x,256,256)
    self.new_act:add(nn.SpatialSymmetricPadding(1,1,1,1))
    self.new_act:add(cudnn.SpatialConvolution(self.from_nfeat,self.from_nfeat,3,3,1,1))

    self.new_act = self.new_act:cuda()
end

function ParallelNet:backward(input_batch, gradOutput, head, ifBranch, useAct)
    local gradInput
    cutorch.setDevice(self.gpu1)
    if head == 1 then
        if useAct then 
            gradInput = self.maskBranch:backward(self.branch_input, gradOutput)
            gradInput = gradInput[1]
            if gradInput == nil then
                print("Backward NIL")
            end
            if ifBranch == false then 
                gradInput = self.trunk:backward(input_batch, gradInput)
            end
        else
            if ifBranch then
                -- print(gradOutput:size())
                -- print(self.branch_input:size())
                gradInput = self.maskBranch:backward(self.branch_input, gradOutput)
            else
                gradInput = self.maskNet:backward(input_batch, gradOutput)
            end
        end
        -- the other forward_prop is ignored
    elseif head == 2 then
        if ifBranch then
            gradInput = self.scoreBranch:backward(self.branch_input, gradOutput)
        else
            gradInput = self.scoreNet:backward(input_batch, gradOutput)
        end
    elseif head == 3 then
        if ifBranch then
            gradInput = self.trunk_head_class:backward(self.branch_input, gradOutput)
        else
            gradInput = self.classNet:backward(input_batch, gradOutput)
        end
    end
    return gradInput
end

function ParallelNet:forward(input_batch,head,iftrain,ifBranch)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch

    if head == 1 then --mask
        self.trunk_out = self.trunk:forward( self.inputs )
        ----- pred is probability of classes
        self.pred = self.softmax:forward(self.trunk_head_class:forward(self.trunk_out))
        self.trunk_head_class:training()
        self.rele_input = self.trunk_head_class:c_backward(self.trunk_out, self.pred)
        self.branch_input = torch.cmul(self.rele_input, self.trunk_out)
        -- Back to original states
        if iftrain == true then
            self.trunk_head_class:training()
        else self.trunk_head_class:evaluate() end

        self.output = self.maskBranch:forward(self.branch_input)
    elseif head == 2 then --score
        if ifBranch then
            self.branch_input = self.trunkNet:forward(self.inputs)
            self.output = self.scoreBranch:forward(self.branch_input)
        else
            self.output = self.scoreNet:forward(self.inputs)
        end
    elseif head == 3 then --class
        if ifBranch then
            --self.branch_input = self.trunkNet:forward(self.inputs)
            --self.output = self.classBranch:forward(self.branch_input)
            self.branch_input = self.trunk:forward(self.inputs)
            self.output = self.trunk_head_class:forward(self.branch_input)
        else
            self.output = self.classNet:forward(self.inputs)
        end
    end

    return self.output
end

return nn.ParallelNet_Mod