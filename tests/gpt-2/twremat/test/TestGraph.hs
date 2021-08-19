{-# Language ScopedTypeVariables #-}
module TestGraph where

import           Control.Monad
import qualified Data.Graph.Inductive.Arbitrary as FGL
import qualified Data.Graph.Inductive.Graph as FGL
import qualified Data.Graph.Inductive.PatriciaTree as FGL
import           Data.List
import           Test.QuickCheck
import           Test.Tasty
import           Test.Tasty.QuickCheck

import           Graph

instance Arbitrary a => Arbitrary (Gr a) where
  arbitrary = do let t = id :: Gen (FGL.Gr a ()) -> Gen (FGL.Gr a ())
                 g <- t arbitrary
                 return (mkGraph (FGL.labNodes g) (FGL.edges g))

newtype TreeOf a = TreeOf { getTreeOf :: Gr a }
  deriving Show

instance Arbitrary a => Arbitrary (TreeOf a) where
  arbitrary = do n <- chooseInt (1, 20)
                 ids <- shuffle [1..n]
                 vals <- replicateM n arbitrary

                 let go tree xs [] = pure tree
                     go tree xs (y:ys) = do
                       x <- elements xs
                       go ((x,y):tree) (y : xs) ys
                 edges <- go [] [head ids] (tail ids)

                 return (TreeOf $ simplify' $ mkGraph (zip ids vals) edges)

newtype DagOf a = DagOf { getDagOf :: Gr a }
  deriving Show

instance Arbitrary a => Arbitrary (DagOf a) where
  arbitrary = do n <- chooseInt (1, 20)
                 ids <- shuffle [1..n]
                 vals <- replicateM n arbitrary

                 let go edges xs [] = pure edges
                     go edges xs (y:ys) = do
                       -- choose an existing node, make the new node a dependency
                       x <- elements xs
                       go ((y,x):edges) (y : xs) ys
                 edges <- go [] [head ids] (tail ids)

                 extra <- case n of
                   1 -> pure []
                   _ -> do
                     n_extra <- chooseInt (0,20)
                     replicateM n_extra $ do
                       sub <- elements (filter ((>1) . length) (tails ids))
                       let (a:b:cs) = sub
                       b <- elements (b:cs)
                       return (b, a)

                 return (DagOf $ mkGraph (zip ids vals) (edges ++ extra))

testGraph :: TestTree
testGraph = testGroup "Graph" [
  testProperty "subgraph nodes" $ \(gr :: Gr ()) ->
      let sub = take (length (nodes gr) `div` 2) (nodes gr)
          subgr = subgraph gr sub
      in nodes (subgr) == sub,
  testProperty "topsort nodes" $ \(gr :: Gr ()) ->
      sort (nodes gr) == sort (topsort gr)
  ]
