import Dataset
import System.IO
import System.Environment
import GHC.Float
import Control.Monad
import Model
import System.Random

dispatch :: String -> [String] -> IO ()
dispatch "new" = new
dispatch "single" = single
dispatch "many" = many
dispatch "verify" = verify
dispatch "epoch" = epoch
dispatch "stats" = stats
dispatch x = (\args -> putStrLn $ (show x) ++ " is not a valid command")

main = do
  args <- getArgs
  if args == [] 
    then do 
      putStrLn "usage: mlp-mnist <subcommand> <args>"
      putStrLn "cli for a multi-layer-perceptron designed to solve MNIST\n"
      putStrLn "subcommands:" 
      putStrLn "  new      create a new model"
      putStrLn "  single   train on a single batch"
      putStrLn "  many     train on many batches in sequence"
      putStrLn "  epoch    train on the entire dataset"
      putStrLn "  verify   check model performance on a single example"
      putStrLn "  stats    get aggregate statistics for model performance"
    else dispatch (head args) (tail args)

new :: [String] -> IO ()
new (path:xs) = saveModel path (rModel (mkStdGen 523) (map read xs)) 
new _ = do 
  putStrLn "mlp-mnist new requires at least 2 arguments"
  putStrLn "usage: mlp-mnist new <path> <dimensions>" 
  putStrLn "create a new multi-layer perceptron\n"
  putStrLn "arguments:"
  putStrLn "  <path>             file to save the new model in"
  putStrLn "  <dimensions>       list of positive integers representing dimensions of the model"
  putStrLn "example:"
  putStrLn "  mlp-mnist new mymodel.mdl 784 16 16 10"
  putStrLn "will create a model that takes in a vector of 784 numbers, and outputs a vector of 10 numbers,\nwith two hidden layers with 16 numbers each, and save it to the file mymodel.mdl"
  putStrLn "\nThe model should have dimensions which begin with 784 and end with 10, to match the MNIST data"

single :: [String] -> IO ()
single (path:start:end:[]) = loadModel path >>= updateOn 0.1 (read start, read end) >>= saveModel path 
single _ = do
  putStrLn "mlp-mnist single requires exactly 3 arguments"
  putStrLn "usage: mlp-mnist single <path> <start> <end>"
  putStrLn "train a model on a single batch\n"
  putStrLn "arguments:"
  putStrLn "  <path>         file containing the model"
  putStrLn "  <start>        index between 0 and 59999 of the first image"
  putStrLn "  <end>          index between 0 and 59999 of the last image"
many :: [String] -> IO ()
many (path:start:end:batchSize:[]) = loadModel path >>= updateSequence 0.1 (read start, read end, read batchSize) >>= saveModel path
many _ = do 
  putStrLn "mlp-mnist many requires exactly 4 arguments"
  putStrLn "usage: mlp-mnist many <path> <start> <end> <batch_size>"
  putStrLn "train a model on a many batches\n"
  putStrLn "arguments:"
  putStrLn "  <path>         file containing the model"
  putStrLn "  <start>        index between 0 and 59999 of the first image"
  putStrLn "  <end>          index between 0 and 59999 of the last image"
  putStrLn "  <batch_size>   number of training examples to put into the same batch"

epoch :: [String] -> IO ()
epoch (path:batchSize:[]) = loadModel path >>= updateSequence 0.1 (0, 60000, read batchSize) >>= saveModel path 
epoch _ = do 
  putStrLn "mlp-mnist epoch requires exactly 2 arguments"
  putStrLn "usage: mlp-mnist epoch <path> <batch_size>"
  putStrLn "train a model on an entire dataset\n"
  putStrLn "arguments:"
  putStrLn "  <path>         file containing the model"
  putStrLn "  <batch_size>   number of training examples to put into the same batch"
  putStrLn "\nMight use all your RAM"

verify :: [String] -> IO ()
verify (path:index:[]) = do
  (image:[]) <- imageR Test (read index, read index + 1)
  (label:[]) <- labelR Test (read index, read index + 1)
  putStrLn $ "Test Set#" ++ index
  putStrLn $ display image
  ((imageInp, labelOutp):[])  <- getBatchData Test (read index, read index + 1)
  putStrLn "Model Guesses:"
  model <- loadModel path
  let output = runModel model imageInp
  putStrLn $ unlines $ (zipWith (\x y -> show x ++ ": " ++ percent y) [0..] output)
  putStrLn $ "Model Guess: " ++ (show $ snd $ foldl1 (max) (zip output [0..]))
  putStrLn $ "Actual Answer: " ++ show label
verify _ = do
  putStrLn "mlp-mnist verify requires exactly 2 arguments"
  putStrLn "usage: mlp-mnist verify <path> <index>"
  putStrLn "check the mode's output on an image in the test set\n"
  putStrLn "arguments:"
  putStrLn "  <path>         file containing the model"
  putStrLn "  <index>        index between 0 and 9999 of image in the test set to check" 

percent :: Double -> String
percent = (++ "%") . take 5 . show . (*100) 

stats :: [String] -> IO ()
stats (path:[]) = do
  testData <- getBatchData Test (0, 10000) 
  model <- loadModel path 
  let raws = map (getStat model) testData
  let accuracy = (int2Double $ length $ filter correct raws)/10000.0
  let avgConfidence = (sum $ map confidence raws)/10000.0
  let avgLoss = (sum $ map eLoss raws)/10000.0
  putStrLn $ "Accuracy: " ++ percent accuracy 
  putStrLn $ "Average Confidence: " ++ percent avgConfidence
  putStrLn $ "Average Loss: " ++ show avgLoss
stats _ = do
  putStrLn "mlp-mnist stats requires exactly 1 argument"
  putStrLn "usage: mlp-mnist stats <path>"
  putStrLn "get overall statistics ifor model's performance in the test set\n"
  putStrLn "arguments:"
  putStrLn "  <path>         file containing the model"

updateOn :: Double -> (Int, Int) -> Model -> IO Model
updateOn rate indicies model = do
  batchData <- getBatchData Train indicies
  let newModel = doBatch rate model batchData
  putStrLn $ "Old Loss: " ++ (show $ loss model (head batchData))
  putStrLn $ "New Loss: " ++ (show $ loss newModel (head batchData))
  putStrLn $ "Did: " ++ show (fst indicies) ++ "-" ++ (show $ snd indicies) ++ "/60000"
  return newModel

updateSequence :: Double -> (Int, Int, Int) -> Model -> IO Model
updateSequence rate (start, end, batchSize) = foldl (>=>) return ops
   where
     ops = map (updateOn rate) indexList
     indexList = map (\x -> (x, x + batchSize)) [start, (start + batchSize)..(end - 1)]
