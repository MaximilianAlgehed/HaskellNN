import Data.List
import Test.QuickCheck

{- TODO:
 - * Implement softmax
 -}

-- Some types
type Delta = Double
type Alpha = Double
type Weights = [Double]
type Bias    = Double
type Parameters = (Weights, Bias)
data Neuron  = Neuron {
                       parameters :: Parameters,
                       neuron :: Parameters -> [Double] -> Double,
                       derivative :: Double -> Double
                      }
type Layer   = [Neuron]
type Network = [Layer]

instance Show Neuron where
    show = show . parameters

weights = fst
bias    = snd

activation :: Parameters 
           -> [Double]
           -> Double
activation (w, b) x = b + sum (zipWith (*) w x)

input :: Parameters
      -> Neuron
input par = Neuron {parameters = par,
                    neuron = \par x -> activation par x,
                    derivative = const 1
                   }

sigmoid :: Parameters
        -> Neuron 
sigmoid par = Neuron {parameters = par,
                      neuron = \par x -> 1 / (1 + exp(negate (activation par x))),
                      derivative = \x -> (exp (negate x)) / (1 + exp (negate x))^2
                     }

relu :: Parameters
     -> Neuron
relu par = Neuron {parameters = par,
                   neuron = \par x -> max 0 (activation par x),
                   derivative = \x -> if x > 0 then 1 else 0
                  }

layer :: Layer
      -> [Double]
      -> [Double]
layer neurons x = map ($x) (map (\n -> neuron n (parameters n)) neurons)

network :: Network
        -> [Double]
        -> [Double]
network net xs = foldl (flip ($)) xs (map layer net)

networkWithIntermidiary :: Network
                        -> [Double]
                        -> [[Double]]
networkWithIntermidiary net xs = scanl (flip ($)) xs (map layer net)

backpropNeuron :: Alpha
               -> Neuron
               -> Delta
               -> [Double] -- Previous input to the neuron
               -> (Neuron, [Delta])
backpropNeuron a neu delt xs = (neu', map (delt*(derivative neu activ)*) (weights (parameters neu))) 
    where
        activ = activation (parameters neu) xs
        outp = neuron neu (parameters neu) xs
        ws   = weights $ parameters neu
        b    = bias $ parameters neu
        errc = a * delt * derivative neu activ
        neu' = neu {parameters = ([w - x*errc | (w, x) <- zip ws xs], b - errc)}

backpropLayer :: Alpha
              -> Layer 
              -> [Delta]
              -> [Double]
              -> (Layer, [Delta])
backpropLayer a lay delt xs = fmap (foldl1 (zipWith (+))) $ unzip $ zipWith (\n d -> backpropNeuron a n d xs) lay delt

backprop :: Alpha
         -> Network
         -> ([Double]  -- Input
            ,[Double]) -- Expected output
         -> Network
backprop a net (inp, out) = reverse
                          $ tail
                          $ map fst
                          $ scanl (\(lay, deltas) (lay', xs) -> backpropLayer a lay' deltas xs) 
                                  ([], delta1)
                                  rnetRes
    where
        delta1 = zipWith (-) (network net inp) (out)
        rnetRes = reverse $ zip net (networkWithIntermidiary net inp)

train :: Alpha
      -> Network
      -> [([Double], [Double])]
      -> Network
train a = foldl (backprop a)

-- Create a network of all 1s
fromConf :: [(Int, Parameters -> Neuron)] -> Gen Network
fromConf xs = sequence [sequence $
                        replicate o $ do
                                        b <- fmap (/100) arbitrary
                                        is <- sequence $ replicate i $ fmap (/100) arbitrary
                                        return $ f (is, b)
                        | ((i, _), (o, f)) <- zip xs (tail xs)] 
