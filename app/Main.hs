module Main where

import qualified  Statistics.Matrix as M
-- import            Data.Vector
import            System.Random
import            Control.Monad.Random



import NeuralNetwork


inputX :: M.Matrix
inputX = M.fromList 3 2 [ 3 , 5
                        , 5 , 1
                        , 10, 2
                        ]

inputY :: M.Matrix
inputY = M.fromList 3 1 [ 75
                        , 82
                        , 93
                        ]


inputXN :: M.Matrix
inputXN = M.map (\x -> x / 24) inputX

inputYN :: M.Matrix
inputYN = M.map (\x -> x / 100) inputY


main :: IO ()
main = do
    let g  = mkStdGen 123123123123
        nn = flip evalRand g $ initNetwork 2 1 3
    print $ costFunction nn inputXN inputYN



main' :: IO ()
main' = do
    let g  = mkStdGen 123123123123123
        nn = flip evalRand g $ initNetwork 2 1 3
        (dJdW1, dJdW2) = costFunctionPrime nn inputX inputY

        scalarW1 = M.fromList 2 3 (replicate 6 3)
        scalarW2 = M.fromList 3 1 (replicate 3 3)

        nn' = nn { weight1Matrix = weight1Matrix nn + (scalarW1 `pointMultiply` dJdW1)
                 , weight2Matrix = weight2Matrix nn + (scalarW2 `pointMultiply` dJdW2) }

    print $ costFunctionPrime nn' inputXN inputYN



g  = mkStdGen 123123123123123
nn = flip evalRand g $ initNetwork 2 1 3
(dJdW1, dJdW2) = costFunctionPrime nn inputXN inputYN

scalarW1 = M.fromList 2 3 (replicate 6 3)
scalarW2 = M.fromList 3 1 (replicate 3 3)


nn' = nn { weight1Matrix = weight1Matrix nn + (scalarW1 `pointMultiply` dJdW1)
         , weight2Matrix = weight2Matrix nn + (scalarW2 `pointMultiply` dJdW2)
         }
