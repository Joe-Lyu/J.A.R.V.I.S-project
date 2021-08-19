{-# Language NamedFieldPuns #-}
{-# Language OverloadedStrings #-}
module Main where

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.List
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Set (Set)
import qualified Data.Set as Set
import           Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import           Debug.Trace
import           System.Environment
import           System.IO
import           Text.Parser.Char
import           Text.Parser.Combinators
import           Text.Trifecta (Parser)
import qualified Text.Trifecta as Trifecta

import           Balanced
import           Filter
import           Graph (Gr, Node)
import qualified Graph as G
import           TWRemat
import           TreeWidth
import           Util

parse :: Parser a -> Text -> a
parse p txt = case Trifecta.parseString p mempty (T.unpack txt) of
  Trifecta.Success a -> a
  Trifecta.Failure e -> error (show (Trifecta._errDoc e))

p_nat :: (Read a, Integral a) => Parser a
p_nat = read <$> some digit

isValidSchedule :: Gr a -> [Step] -> [Node] -> Bool
isValidSchedule gr steps ts = go steps IS.empty
  where
    go (Compute n : steps) live = all (`IS.member` live) (G.preList gr n) && go steps (IS.insert n live)
    go (Free n : steps) live = IS.member n live && go steps (IS.delete n live)
    go [] live = all (\n -> IS.member n live) ts

-- Modify the graph to insert direct dependencies through 'pointer'
-- type ops to the op that generated their underlying
-- storage. Example, given op1 -> id -> op2, op2 will now directly
-- keep op1 alive. Assuming that 'pointer' type ops are ~0 cost the
-- 'id' can now be immediately freed after use, and all memory usage
-- charged to op1, simplifying memory analysis.
mergePointers :: Gr a -> (Node -> Weight) -> Gr a
mergePointers gr info = merged
  where
    merged = G.insEdges [(p, n) | n <- G.nodes gr, p <- pdeps n] gr
    pdeps n = let pparents n = [p | p <- G.preList gr n, info p == Pointer]
                  go [] visited = Set.toList visited
                  go (p:ps) visited | Set.member p visited = go ps visited
                                    | otherwise = go (pparents p ++ ps) (Set.insert p visited)
              in go (pparents n) Set.empty


outputSchedule :: [Step] -> IO ()
outputSchedule schedule = do
  args <- getArgs
  let printStep (Compute n) = "c " <> T.pack (show n)
      printStep (Free n) = "f " <> T.pack (show n)
      output = T.unlines (map printStep schedule)
  case args of
    [path, outpath] -> T.writeFile outpath output
    [path] -> T.putStr output


main :: IO ()
main = do
  args <- getArgs
  let path = head args
  txt <- T.readFile path
  let p = do
        string "p remat2" *> spaces
        memlimit <- optional (text "memlimit" *> spaces *> p_nat <* spaces)
        nodes <- some $ do
          text "node" <* spaces
          node_id <- p_nat <* spaces
          deps <- fold <$> optional (text "deps" *> spaces *> many (p_nat <* spaces))
          let p_weight = do
                cpu <- text "cpu" *> spaces *> p_nat <* spaces
                mem <- text "mem" *> spaces *> p_nat <* spaces
                return (Normal{cpu,mem})
              p_effectful = const Effectful <$> text "effectful" <* spaces
              p_pointer = const Pointer <$> text "pointer" <* spaces
          weight <- optional (p_weight <|> p_effectful <|> p_pointer)
          target <- (const True <$> text "target" <* spaces) <|> pure False
          optional (char '\n')
          return (node_id, deps, weight, target)
        eof
        return (memlimit, nodes)
      (memlimit, node_data) = parse p txt
      ns = [n | (n, _, _, _) <- node_data]
      es = [(d, n) | (n, ds, _, _) <- node_data, d <- ds]
      ts = [n | (n, _, _, True) <- node_data]

  case memlimit of
    Just memlimit -> let
      weights = Map.fromList [(n,w) | (n, _, Just w, _) <- node_data]
      weight n = Map.findWithDefault (Normal 1 1) n weights
      graph = mergePointers (G.mkUGraph ns es) weight
      schedule = remat graph (IS.fromList ts)
      schedule' = optimize graph weight memlimit schedule
      in do outputSchedule schedule'
            hPutStrLn stderr ("isValid = " ++ show (isValidSchedule graph schedule' ts))
            hPutStrLn stderr ("length = " ++ show (length schedule'))
            evalSched weight (initSched graph schedule')
    Nothing -> let
      graph = G.mkUGraph ns es
      schedule = remat graph (IS.fromList ts)
      in outputSchedule schedule
  -- -- G.plotLab "tree.dot" (IS.toList <$> treeWidth graph)
  -- print (length ns)
  -- print (length schedule, isValidSchedule graph schedule ts)
  -- print (length schedule', isValidSchedule graph schedule' ts)

  -- let sched_1 = initSched graph schedule
  -- evalSched weight sched_1

  -- let sched_2 = optSched sched_1
  -- evalSched weight sched_2

  -- let go sched = do
  --       let sched_1 = greedy weight 1000 sched
  --       evalSched weight sched_1
  --       let sched_2 = optSched sched_1
  --       evalSched weight sched_2
  --       go sched_2

  -- go sched_2

  -- let
  --   output = T.unlines $ map T.pack $ do
  --     step <- schedule
  --     return $ case step of
  --       Compute n -> "c " ++ show n
  --       Free n -> "f " ++ show n
  -- case args of
  --   [path, outpath] -> T.writeFile outpath output
  --   [path] -> T.putStr output
  -- hPutStrLn stderr ("isValid = " ++ show (isValidSchedule graph schedule))
