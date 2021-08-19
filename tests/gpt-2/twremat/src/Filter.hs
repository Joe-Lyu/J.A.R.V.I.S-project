{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Filter where

import           Control.Monad.State.Lazy
import           Data.Foldable
import           Data.List
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Monoid
import           Data.OrdPSQ (OrdPSQ)
import qualified Data.OrdPSQ as PSQ
import           Data.Relation (Relation)
import qualified Data.Relation as R
import           Data.Semigroup
import           Data.Set (Set)
import qualified Data.Set as Set
import           Debug.Trace

import           Graph (Gr, Node)
import qualified Graph as G
import           TWRemat

newtype CID = CID Int
  deriving (Show, Eq, Ord)


-- Indexed data structure for rematerializion schedule.
data Sched = Sched {
  computes :: Relation CID Node, -- compute step -> node id (many-1)
  c_free :: Relation CID Node, -- compute step -> nodes freed afterward (many-many)
  c_require :: Relation CID Node -- compute step -> nodes required as input (many-many)
  }
  deriving (Show)

pattern One :: a -> Set a
pattern One a <- (Set.toList -> [a]) where
  One a = Set.singleton a

pattern None :: Set a
pattern None <- (Set.null -> True) where
  None = Set.empty

deleteL :: (Ord a, Ord b) => a -> Relation a b -> Relation a b
deleteL a rel = foldr go rel (Set.toList $ R.lookupDom a rel)
  where
    go b = R.delete a b

-- Info for each node relevant to evaluating and optimizing cpu/memory
-- consumption.
data Weight =
  -- Normal node that reads its inputs and produces some output.
  Normal{cpu::Int, mem::Int}
  -- Effectful node that must not be duplicated (eg. assigning to a
  -- variable). Assumed to produce no relevant output.
  | Effectful
  -- Pointer nodes return a view that shares memory with dependencies,
  -- keeping them from being GCd (example: tf.identity).
  | Pointer
  deriving (Eq, Ord)

-- Reduce cpu usage of a schedule under the constraint that peak mem usage must be less than `memLimit`.
greedy :: (Node -> Weight) -> Int -> Sched -> Sched
greedy info memLimit sched@Sched{computes,c_free,c_require} = go (R.toList computes) Set.empty 0 Set.empty PSQ.empty
  where
    memOf n = case info n of
      Normal{mem} -> mem
      Effectful -> 0
      Pointer -> 0

    priority :: CID -> Node -> Maybe Double
    priority c n
      -- Anything we don't need anymore should be freed immediately.
      | Nothing == Set.lookupGT c (R.lookupRan n c_require) = Just 0
      | otherwise = case info n of
          -- Otherwise, prioritise the ops which use most memory and least cpu to recompute.
          Normal{mem, cpu} -> Just (fromIntegral cpu / fromIntegral mem)
          -- Effectful ops are assumed to use no memory and should never be freed.
          Effectful -> Nothing
          -- Free pointer ops immediately.
          Pointer -> Just 0

    go [] keepcid memUsage live freeList =
      let finish c Sched{computes,c_free,c_require} = case R.lookupDom c computes of
            One n -> Sched{computes = R.delete c n computes,
                           c_free = case (Set.lookupLT c (R.lookupRan n computes),
                                          Set.lookupLT c (R.lookupRan n c_free)) of
                                      (Just cc, Just fc) | cc <= fc -> R.delete fc n c_free,
                           c_require = deleteL c c_require}
      in foldr (.) id [finish c | c <- toList (R.dom computes), not (Set.member c keepcid)] sched
    go ((c,n):cs) keepcid memUsage live freeList
      | Set.member n live = go_free c cs keepcid memUsage live (PSQ.delete n freeList)
      | memUsage + memOf n > memLimit && not (PSQ.null freeList) = case PSQ.findMin freeList of
          Just (f,_,_) ->
            go ((c,n):cs) keepcid (memUsage - memOf f) (Set.delete f live) (PSQ.deleteMin freeList)
      | otherwise = go_next (c,n) cs keepcid memUsage live freeList

    go_next (c,n) cs keepcid memUsage live freeList =
      go_free c cs (Set.insert c keepcid) (memUsage + memOf n) (Set.insert n live) (PSQ.delete n freeList)

    go_free c cs keepcid memUsage live freeList =
      let freeList' = foldr (.) id [PSQ.insert n v ()
                                   | n <- Set.toList (R.lookupDom c c_free),
                                     Just v <- [priority c n]] freeList
      in go cs keepcid memUsage live freeList'

evalSched :: (Node -> Weight) -> Sched -> IO ()
evalSched info Sched{computes,c_free,c_require} = do
  putStrLn (unwords ["steps=" ++ show (Set.size (R.dom computes)),
                     "cpu=" ++ show (sum [cpu | (c, n) <- R.toList computes, Normal{cpu} <- [info n]]),
                     "peak=" ++ show peak])
  where
    memOf n = case info n of
      Normal{mem} -> mem
      _ -> 0
    peak = go (Set.toList (R.dom computes <> R.dom c_free)) 0 0
    go [] maxMem curMem = maxMem
    go (c:cs) maxMem curMem = case R.lookupDom c computes of
      One n -> go cs (max maxMem (curMem + memOf n)) (curMem + memOf n - sum [memOf f | f <- toList (R.lookupDom c c_free)])
      None -> go cs maxMem (curMem - sum [memOf f | f <- toList (R.lookupDom c c_free)])

-- Basic optimizations: move Free actions to be as early as possible,
-- and eliminate Compute actions that are immediately Freed without
-- being used.
optSched :: Sched -> Sched
optSched sched@Sched{computes, c_free} = foldr go sched (toList (R.dom computes <> R.dom c_free))
  where
    checkAnnihilate c sched@Sched{..} =
      case R.lookupDom c computes of
        One n | R.member c n c_free ->
                Sched{computes = R.delete c n computes,
                      c_free = R.delete c n c_free,
                      c_require = deleteL c c_require}
        _ -> sched
    checkMove c sched@Sched{..} =
      case R.lookupDom c c_free of
        ns | Set.size ns > 0 ->
             let target n = getMax <$> fold [Max <$> Set.lookupLE c (R.lookupRan n computes),
                                             Max <$> Set.lookupLE c (R.lookupRan n c_require),
                                             Max <$> Set.lookupLT c (R.lookupRan n c_free)]
                 process n sched@Sched{..} = case target n of
                   Just c' | c' < c -> sched { c_free = R.insert c' n $ R.delete c n $ c_free }
                   Just c' | c'== c -> sched
                   Nothing -> sched { c_free = R.delete c n $ c_free }
             in foldr process sched (Set.toList ns)
        _ -> sched

    go c sched = checkMove c $ checkAnnihilate c sched


initSched :: Gr a -> [Step] -> Sched
initSched gr sched = Sched{computes, c_free, c_require}
  where
    steps = Map.fromList (zip [1..] sched) :: Map Int Step
    computes = R.fromList [(CID k, n) | (k, Compute n) <- Map.toList steps]
    cdom = R.dom computes
    c_free = R.fromList [let Just c = Set.lookupLT (CID k) cdom
                         in (c, n) | (k, Free n) <- Map.toList steps]
    c_require = R.fromList [(c, p) | (c, n) <- R.toList computes, p <- G.preList gr n]

runSched :: Sched -> [Step]
runSched Sched{computes, c_free} = fold [[Compute n] ++ (Free <$> Set.toList (R.lookupDom c c_free))
                                        | (c, n) <- R.toList computes]

-- 6 cycles of forward-backward optimization seems to generally be enough for a good schedule.
optimize :: Gr a -> (Node -> Weight) -> Int -> [Step] -> [Step]
optimize gr info memLimit steps = runSched (foldl' step startSched [1..maxSteps])
  where
    step !sched i = trace ("Optimizing schedule... " ++ show i ++ "/" ++ show maxSteps) $ optSched (greedy info memLimit sched)
    startSched = optSched (initSched gr steps)
    maxSteps = 6
