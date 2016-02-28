{-# LANGUAGE RecordWildCards, BangPatterns #-}

module NeuralNetwork where

import qualified  Statistics.Matrix as M
import            Control.Monad.Random


-- hyperparameters

data NeuralNetwork =
     NeuralNetwork { inputLayerSize  :: Int
                   , outputLayerSize :: Int
                   , hiddenLayerSize :: Int
                   , weight1Matrix   :: M.Matrix
                   , weight2Matrix   :: M.Matrix
                   }


initNetwork :: MonadRandom r => Int -> Int -> Int -> r NeuralNetwork
initNetwork inputSize outputSize hiddenSize = do
    rs  <- getRandomRs (0.0, 1.0)
    rs' <- getRandomRs (0.0, 1.0)
    return $ NeuralNetwork { inputLayerSize  = inputSize
                           , outputLayerSize = outputSize
                           , hiddenLayerSize = hiddenSize
                           , weight1Matrix   = M.fromList inputSize  hiddenSize $ take (inputSize  * hiddenSize) rs
                           , weight2Matrix   = M.fromList hiddenSize outputSize $ take (outputSize * hiddenSize) rs'
                           }

forwardRs :: NeuralNetwork -> M.Matrix -> (M.Matrix, M.Matrix, M.Matrix, M.Matrix)
forwardRs NeuralNetwork{..} matrix =
    let !z2 = matrix * weight1Matrix
        !a2 = M.map sigmoid z2
        !z3 = a2 * weight2Matrix
    in  (M.map sigmoid z3, z2, a2, z3)

forward :: NeuralNetwork -> M.Matrix -> M.Matrix
forward nn matrix =
    let (r, _, _, _) = forwardRs nn matrix
    in r


sigmoid :: Double -> Double
sigmoid z = 1 / (1 + exp (-z))

-- | Derivative of sigmoid function
sigmoidPrime :: Double -> Double
sigmoidPrime z = ez / ((1 + ez) ** 2)
    where ez = exp (-z)

costFunction :: NeuralNetwork -> M.Matrix -> M.Matrix -> Double
costFunction nn@NeuralNetwork{..} x y =
    let yHat = nn `forward` x
        err  = sum . map (**2) . M.toList $ y - yHat
    in  0.5 * err


costFunctionPrime :: NeuralNetwork -> M.Matrix -> M.Matrix -> (M.Matrix, M.Matrix)
costFunctionPrime nn@NeuralNetwork{..} x y = 
    let (yHat, z2, a2, z3) = nn `forwardRs` x
        !delta3            =  (- (y - yHat)) `pointMultiply` (M.map sigmoidPrime z3)
        !dJdW2             = (M.transpose a2) * delta3

        !delta2            = delta3 * (M.transpose weight2Matrix) * (M.map sigmoidPrime z2)
        !dJdW1             = M.transpose x * delta2

    in (dJdW1, dJdW2)


pointMultiply :: M.Matrix -> M.Matrix -> M.Matrix
pointMultiply m1 m2 =
    let (r, c) = (M.rows m1, M.cols m1)
        (l1, l2) = (M.toList m1, M.toList m2)
    in M.fromList r c (foldr (\(a,b) xs -> (a * b) : xs) [] (zip l1 l2))


instance Num M.Matrix where
    m1 + m2  = let (r, c) = (M.rows m1, M.cols m1)
                   (l1, l2) = (M.toList m1, M.toList m2)
               in M.fromList r c (foldr (\(a,b) xs -> (a + b) : xs) [] (zip l1 l2))

    negate m      = let (r, c) = (M.rows m, M.cols m)
                        l      = M.toList m
                    in M.fromList r c (map negate l)

    m1 * m2  = m1 `M.multiply` m2

    abs m         = undefined
    signum m      = undefined
    fromInteger m = undefined