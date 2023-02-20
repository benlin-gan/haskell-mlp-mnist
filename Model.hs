module Model
( rModel
, Weight
, Bias 
, Model
, weights
, biases
, Input
, Output
, saveModel
, loadModel
, runModel
, doBatch
, loss
, Stat
, getStat
, correct
, confidence
, eLoss
) where
import System.Random
import Mat
import Dataset
import Control.Monad
import Control.Monad.Writer
import GHC.Float
import System.IO
import System.Directory
import Control.Exception

type Schema = [Int]
type Weight = Mat
type Bias = [Double]
data Model = Model {
  weights :: [Weight], 
  biases :: [Bias]
} deriving (Show, Read, Eq)

saveModel :: FilePath -> Model -> IO ()
saveModel path model = bracketOnError (openTempFile "." "temp") 
  (\(tempName, tempHandle) -> do
    hClose tempHandle
    removeFile tempName)
  (\(tempName, tempHandle) -> do
    hPutStr tempHandle (show model)
    hClose tempHandle
    renameFile tempName path) 
loadModel :: FilePath -> IO Model
loadModel f = readFile' f >>= return . read 
 
pprintM :: Model -> String
pprintM m = "Weights:\n" ++ unlines (map pprint (weights m)) ++ "Biases:\n" ++ pprint (biases m)

addM :: Model -> Model -> Model
addM a b = Model { weights = newW, biases = newB} 
  where
    newW = zipWith (add2D) (weights a) (weights b) 
    newB = add2D (biases a) (biases b)

scaleM :: Double -> Model -> Model
scaleM s m = Model { weights = newW, biases = newB}
  where
    newW = map (scale s) (weights m)
    newB = scale s (biases m) 

getSchema :: Model -> Schema
getSchema m = (length $ head $ head (weights m)):(map length (biases m))

rWeights :: RandomGen g => g -> Schema -> ([Weight], g)
rWeights g [] = ([], g)
rWeights g [x] = ([], g)
rWeights g (x:(y:rest)) = (a:(fst $ rWeights ng (y:rest)), ng)
  where (a, ng) = rMat g y x

rBiases :: RandomGen g => g -> Schema -> ([Bias], g)
rBiases g [] = ([], g)
rBiases g [x] = ([], g)
rBiases g (x:(y:rest)) = (a:(fst $ rBiases ng (y:rest)), ng)
  where (a, ng) = rVec g y 

rModel :: RandomGen g => g -> Schema -> Model
rModel g s = Model { weights = a, biases = fst $ rBiases ng s}
  where (a, ng) = rWeights g s

logR :: a -> Writer [a] a
logR x = writer (x, [x]) 

unloggedF :: (a -> a) -> a -> Writer [a] a
unloggedF f x = writer (f x, [])

loggedF :: (a -> a) -> a -> Writer [a] a
loggedF f x = unloggedF f x >>= logR  

type Intermediate = [Double]

e :: Double 
e = 2.71828182846

sig :: Double -> Double  
sig x = 1/(1 + e ** (-x))

forwardTransfa :: [Weight] -> [Intermediate -> Writer [Intermediate] Intermediate] 
forwardTransfa ws = unloggedF <$> (map mvmul ws)

forwardTransfb :: [Bias] -> [Intermediate -> Writer [Intermediate] Intermediate]
forwardTransfb bs = loggedF <$> map (zipWith (+)) bs 

forwardTransfc :: [Bias] -> [Intermediate -> Writer [Intermediate] Intermediate]
forwardTransfc bs = unloggedF <$> [map sig | _ <- bs]  

trtl :: ((a, a), a) -> [a]
trtl ((x, y), z) = [x, y, z]

dtl :: (a, a) -> [a]
dtl (x, y) = [x, y]

forwardTransf :: Model -> Intermediate -> Writer [Intermediate] Intermediate  
forwardTransf m = foldl (>=>) return operations 
  where 
    operations :: [Intermediate -> Writer [Intermediate] Intermediate]
    operations = concat $ map trtl (zip (zip as bs) cs) 
    as = forwardTransfa (weights m)
    bs = forwardTransfb (biases m)
    cs = forwardTransfc (biases m)

backwardTransfa :: [Weight] -> [Intermediate -> Writer [Intermediate] Intermediate]
backwardTransfa ws = unloggedF <$> map tmvmul (tail ws) 

sigd :: Double -> Double
sigd x = sig x * (1 - sig x) 

type Winput = [Double]
type Dwinput = [Double]

derivedWinputs :: [Winput] -> [Dwinput]
derivedWinputs = map $ map sigd  

activatedWinputs :: [Winput] -> [Intermediate]
activatedWinputs = map $ map sig

backwardTransfb :: [Winput] -> [Intermediate -> Writer [Intermediate] Intermediate]
backwardTransfb winputs = loggedF <$> map (zipWith (*)) (derivedWinputs winputs)  

intersperse :: [a] -> [a] -> [a] 
intersperse (x:xs) [] = [x] 
intersperse [] _ = [] 
intersperse (x:xs) (y:ys) = x:y:(intersperse xs ys)

backwardTransf :: Model -> [Intermediate] -> Intermediate -> Writer [Intermediate] Intermediate
backwardTransf m i = foldl (<=<) return (intersperse b a)
  where 
    a = backwardTransfa (weights m)
    b = backwardTransfb i

type Input = [Double]
type Output = [Double]
type WinputUpdate = [Intermediate]
type Update = Model

makeUpdate :: WinputUpdate -> [Intermediate] -> Update
makeUpdate wiu prevAct = Model {weights = newW, biases = wiu } 
  where newW = zipWith (oProd) prevAct wiu 

train :: Model -> (Input, Output) -> Update 
train m (i, o) = makeUpdate (reverse winputUpdates) prevAct 
  where 
    prevAct = i:(activatedWinputs winputs)
    ( _ , winputUpdates) = runWriter $ backwardTransf m winputs lastWinputUpdate 
    lastWinputUpdate = map (*2) (zipWith (-) guess o)
    (guess, winputs) = runWriter $ forwardTransf m i

batch :: Model -> [(Input, Output)] -> Update 
batch m io = foldl (addM) (scaleM 0 m) (map (train m) io)

doBatch :: Double -> Model -> [(Input, Output)] -> Model
doBatch rate m io = addM m (scaleM factor (batch m io))
  where
    factor = -(rate/(int2Double $ length io))

loss :: Model -> (Input, Output) -> Double
loss m (i, o) = sum $ zipWith (*) diffs diffs 
  where 
    diffs = (zipWith (-) guess o)
    (guess, _) = runWriter $ forwardTransf m i

runModel :: Model -> Input -> Output
runModel m i = fst $ runWriter $ forwardTransf m i

data Stat = Stat {
  correct :: Bool,
  confidence :: Double, 
  eLoss :: Double
}
interpret :: [Double] -> (Double, Int)
interpret v = foldl1 (max) (zip v [0..])

getStat :: Model -> (Input, Output) -> Stat
getStat m (i, o) = Stat {correct = c, confidence = d, eLoss = loss m (i, o)} 
  where
   c = actual == guess 
   (_, actual) = interpret o
   (d, guess) = interpret (runModel m i) 
