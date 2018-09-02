local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function copy_core(Net,list)
    local all = {}
    --print(Net)
    for i,k in pairs(list) do
        --print(Net[k])
        all[k] = deepCopy(Net[k]):float():clearState()
    end
    return all
end
