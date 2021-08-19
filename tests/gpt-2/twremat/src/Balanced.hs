{-# Language BangPatterns #-}
{-# Language DeriveTraversable #-}
module Balanced where

import           Data.Bifunctor
import           Data.Foldable
import           Data.IntMap (IntMap)
import qualified Data.IntMap as IM
import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Data.List
import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.Ord
import           Data.Set (Set)
import qualified Data.Set as Set
import           Data.Tuple
import           Debug.Trace

import           TreeWidth

import           Graph (Gr, Node)
import qualified Graph as G
import           Util

balancedSeparator :: Gr a -> Node
balancedSeparator gr = minimumOn (\n -> (weight n, n)) (G.nodes gr)
  where
    cutWeight = memo (G.edges gr) $ \(a,b) ->
      1 + sum [cutWeight (b,c) | c <- G.sucList gr b, c /= a] :: Int
    weight = \a ->
      maximum [cutWeight (a,b) | b <- G.sucList gr a]

-- Rose tree with weight annotations.
data Tree a = Tree Int a [Tree a]
  deriving (Show, Functor, Foldable, Traversable)

treeWeight :: Tree a -> Int
treeWeight (Tree w _ _) = w

treeVal :: Tree a -> a
treeVal (Tree _ a _) = a

tree :: a -> [Tree a] -> Tree a
tree a subs = Tree (1 + sum (map treeWeight subs)) a subs

-- Create a Tree from treelike graph. Assumes gr is undirected and
-- simple and has at least one node.
mkTree :: Gr a -> Tree Node
mkTree gr = tree top [go top v | v <- G.sucList gr top, v /= top]
  where
    go u v = tree v [go v w | w <- G.sucList gr v, w /= u]
    top = head (G.nodes gr)

-- Choose one element from a list.
choose1 :: [a] -> [(a, [a])]
choose1 xs = do i <- [0..length xs-1]
                return (xs!!i, take i xs ++ drop (i+1) xs)

-- Balance a tree by recursively rotating each node until the heaviest
-- subtree has minimal weight. The result is a tree with two
-- properties:
--
-- 1. For every node v in the tree, the subtrees rooted at children of
-- v are disjoint connected components of the original tree with v
-- removed.
--
-- 2. The tree is balanced, in that for every node v, the heaviest
-- child of v has weight at most weight[v]/2.

balance :: Tree a -> Tree a
balance root@(Tree x a []) = root
balance root@(Tree x a children)
  -- If we can improve balance by rotating, do so and check again.
  | bestscore < score root = balance best
  -- Current level is balanced, now balance all children.
  | otherwise = tree a (map balance children)
  where
    rotate (Tree _ new_a new_children) other_children =
      let old_root = tree a other_children
      in tree new_a (old_root : new_children)
    options = [rotate choice rest | (choice, rest) <- choose1 children]
    score (Tree x a children)
      | null children = 0
      | otherwise = maximum (map treeWeight children)
    (bestscore, best) = minimumOn fst [(score t, t) | t <- options]

-- Convert a
sepTree :: Gr a -> Tree a
sepTree gr = fmap (G.lab gr) $ balance $ mkTree gr

-- sepTreeSlow :: Gr a -> Tree a
-- sepTreeSlow gr
--   | G.order gr == 1 = tree (snd $ head $ G.labNodes gr) []
--   | otherwise = tree (G.lab gr top) [sepTreeSlow sub | sub <- G.splitComponents (G.delNode top gr)]
--       where
--         top = balancedSeparator gr
