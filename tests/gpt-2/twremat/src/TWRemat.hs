{-# Language BangPatterns #-}
{-# Language DeriveTraversable #-}
{-# Language NamedFieldPuns #-}
module TWRemat where

import           Data.DList (DList)
import qualified Data.DList as DL
import           Data.Foldable
import           Data.IntMap (IntMap)
import qualified Data.IntMap as IM
import           Data.Map (Map)
import qualified Data.Map.Lazy as Map
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.List
import           Data.Ord

import           Balanced
import           Graph (Gr, Node)
import qualified Graph as G
import           TreeWidth
import           Util

data Step = Compute Node | Free Node
  deriving (Show, Eq, Ord)

toposort :: Gr a -> IntSet -> [Node]
toposort gr = go
  where
    go xs = sortOn (\x -> score IM.! x) (IS.toList xs)
    score = IM.fromList (zip (G.topsort gr) [0..])

ancestors :: Gr a -> (Node -> IntSet)
ancestors gr = tab
  where
    tab = memo (G.nodes gr) (\n -> IS.singleton n <> foldMap tab (G.preList gr n))

-- Recursively remove elements of root node from all subtrees.
preFilter :: Tree Bag -> Tree Bag
preFilter (Tree _ x subs) = tree x [preFilter $ fmap (`IS.difference` x) c | c <- subs]

data Comp = Comp{x :: Bag, xall :: Bag}

-- Annotate each node with the union of all nodes in its subtree.
preFold :: Tree Bag -> Tree Comp
preFold t@(Tree _ x subs) = tree Comp{x,xall=total} subs'
  where subs' = preFold <$> subs
        total = x <> fold (xall . treeVal <$> subs')

-- Computes a rematerialization schedule for the given DAG, which ends
-- with the nodes of 'compute' computed and in memory.
remat :: Gr a -> IntSet -> [Step]
remat gr compute = DL.toList (twremat (preFold . preFilter . sepTree $ treeWidth gr) compute)
  where
    topo = toposort gr
    antab = ancestors gr

    twremat :: Tree Comp -> IntSet -> DList Step
    twremat (Tree _ Comp{x} components) compute
      | IS.null compute = mempty
      | otherwise = case components of
          [] ->
            -- Base case: simply execute the remaining nodes in order, then
            -- free the ones the caller doesn't need.
            DL.fromList (Compute <$> topo target) <> DL.fromList (Free <$> topo (target `IS.difference` compute))
          components ->
            -- Recursion case: select a balanced separator X of the tree decomposition.
            -- 1. for each node v of X that we need to compute, in topological order
            --   a. Recursively compute the direct dependencies of v in each subtree,
            --      excluding any which are in X itself (those are already computed
            --      and in memory, since we are traversing X in topological order).
            --   b. Compute v.
            --   c. Free the dependencies computed in #1a.
            -- 2. Recursively compute the needed nodes which are not in X
            -- 3. Free the computed nodes of X that the caller doesn't need.
            let compsets = map (xall . treeVal) components :: [Bag]
                part1 v = let deps = G.pre gr v
                              new_computes = [deps `IS.intersection` chi_nodes | chi_nodes <- compsets]
                          in fold [twremat chi new_compute | (chi, new_compute) <- zip components new_computes]
                             <> (DL.singleton (Compute v))
                             <> DL.fromList (Free <$> (IS.toList $ fold new_computes))
                part2 = fold [twremat chi (outside `IS.intersection` chi_nodes) | (chi, chi_nodes) <- zip components compsets]
                part3 = DL.fromList (Free <$> topo (target `IS.difference` compute))
            in foldMap part1 (topo target) <> part2 <> part3
      where
        ancestor_set = foldMap antab (IS.toList compute) :: IntSet
        -- Nodes of X which are needed, directly or indirectly.
        target = IS.filter (\i -> IS.member i ancestor_set) x
        -- Nodes the caller needs which are not in X.
        outside = compute `IS.difference` x
