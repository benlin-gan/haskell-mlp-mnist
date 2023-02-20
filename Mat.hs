module Mat
( rMat
, rVec
, pprint
, Mat
, mvmul
, tmvmul
, oProd
, add2D
, scale
) where
import System.Random
import Data.List

type Mat = [[Double]]

pprint :: Mat -> String
pprint = unlines . map show

rVec :: RandomGen g => g -> Int -> ([Double], g)
rVec g 0 = ([], g)
rVec g n = (x:(fst $ rVec ng (n - 1)), ng)
  where (x, ng) = randomR (-1.0, 1.0) g

rMat :: RandomGen g => g -> Int -> Int -> (Mat, g)
rMat g 0 _ = ([], g)
rMat g r c = (x:(fst $ rMat ng (r-1) c), ng)
  where (x, ng) = rVec g c

zeroes :: Int -> Int -> Mat
zeroes r c = take r $ repeat $ take c $ (repeat 0)

dot ::  [Double] -> [Double] -> Double
dot x y = sum $ zipWith (*) x y

vmul :: [Double] -> Mat -> [Double]
vmul v = map (dot v)

mvmul :: Mat -> [Double] -> [Double]
mvmul = flip vmul

tmvmul :: Mat -> [Double] -> [Double]
tmvmul = mvmul . transpose 

oProd :: [Double] -> [Double] -> Mat
oProd rs cs = map ($ rs) (map map (map (*) cs))

add2D :: (Num a) => [[a]] -> [[a]] -> [[a]]
add2D _ [] = []
add2D [] _ = []
add2D (x:xs) (y:ys) = (zipWith (+) x y):(add2D xs ys)   

scale :: Double -> Mat -> Mat
scale s = map (map (*s)) 
