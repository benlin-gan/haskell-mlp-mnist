module Dataset 
( getBatchData
, display
, imageR
, labelR
, group
, DataMode (..)
) where
import System.IO
import Control.Monad.Writer
import Data.Char
import GHC.Float
import GHC.Num.Integer

data DataMode = Test | Train

labelR :: DataMode -> (Int, Int) -> IO [Int]
labelR dataMode (start, end) = withBinaryFile (file dataMode) ReadMode ( \handle -> do
  hSeek handle AbsoluteSeek (8 + integerFromInt start) 
  s <- hGetChars (end - start) handle
  return $ map ord s )
    where
      file :: DataMode -> String
      file Test = "t10k-labels-idx1-ubyte" 
      file Train = "train-labels-idx1-ubyte"

onehot :: Int -> [Double]
onehot x = (replicate (x) 0.0) ++ [1.0] ++ (replicate (9 - x) 0.0)

hGetChars :: Int -> Handle -> IO String
hGetChars 0 _ = return ""
hGetChars n h = (:) <$> (hGetChar h) <*> (hGetChars (n-1) h)

hGetImg :: Handle -> IO Image
hGetImg h = (group 28 28) <$> (map ord) <$> hGetChars 784 h

hGetImgs :: Int -> Handle -> IO [Image]
hGetImgs 0 _ = return []
hGetImgs n h = (:) <$> (hGetImg h) <*> (hGetImgs (n-1) h)

imageR :: DataMode -> (Int, Int) -> IO [Image]
imageR dataMode (start, end) = withBinaryFile (file dataMode) ReadMode ( \handle -> do
  hSeek handle AbsoluteSeek (16 + 784 * integerFromInt start)
  hGetImgs (end - start) handle )
    where
      file :: DataMode -> String
      file Test = "t10k-images-idx3-ubyte" 
      file Train = "train-images-idx3-ubyte"

getBatchData :: DataMode -> (Int, Int) -> IO [([Double], [Double])]
getBatchData dataMode indicies = do
  is <- imageR dataMode indicies
  ls <- labelR dataMode indicies
  let cis = map compress is
  let cls = map onehot ls
  return $ zip cis cls


type Image = [[Int]]
pixel :: Int -> String
pixel x
  | x < 32 = "   "
  | x < 64 = "..."
  | x < 96 = "..:"
  | x < 128 = "..|" 
  | x < 160 = ".:|"
  | x < 192 = ".||"
  | x < 224 = ":||"
  | otherwise = "|||" 
display :: Image -> String
display x = unlines $ map concat (map (map (pixel)) x)

compress :: Image -> [Double] 
compress = (map (/256.0)) . map (int2Double) . concat

group :: Int -> Int -> [a] -> [[a]]
group r c xs = execWriter $ (foldl (>=>) (return) (replicate r (extract c))) xs  

extract :: Int -> [a] -> Writer [[a]] [a]
extract n xs = writer (d, [t]) 
  where 
    (t, d) = splitAt n xs
