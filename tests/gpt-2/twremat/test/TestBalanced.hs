module TestBalanced where

import           Control.Monad
import           Data.Foldable
import           Test.QuickCheck
import           Test.Tasty
import           Test.Tasty.QuickCheck

import           Balanced
import           Graph (Gr, Node)
import qualified Graph as G
import           TestGraph

subTrees :: Tree a -> [Tree a]
subTrees t@(Tree w a []) = [t]
subTrees t@(Tree w a cs) = t : foldMap subTrees cs

testBalanced :: TestTree
testBalanced = testGroup "Balanced" [
  testProperty "Subtrees are connected" $ \(TreeOf gr) ->
      let t = mkTree (gr :: Gr ())
          go tree = G.isConnected (G.subgraph gr (toList tree))
      in all go (subTrees t),
  testProperty "Subtrees are connected after balance" $ \(TreeOf gr) ->
      let t = balance $ mkTree (gr :: Gr ())
          go tree = G.isConnected (G.subgraph gr (toList tree))
      in all go (subTrees t),
  testProperty "Subtrees are balanced" $ \(TreeOf gr) ->
      let t = balance $ mkTree (gr :: Gr ())
          go t@(Tree w a []) = True
          go t@(Tree w a cs) = maximum (map treeWeight cs) <= div w 2
      in all go (subTrees t)
  ]
