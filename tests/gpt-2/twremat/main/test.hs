{-# Language ScopedTypeVariables #-}
module Main where

import           Data.IntSet (IntSet)
import qualified Data.IntSet as IS
import           Test.QuickCheck
import           Test.Tasty
import           Test.Tasty.QuickCheck

import           Graph
import           TWRemat
import           TestBalanced
import           TestGraph
import           TestTreeWidth

isValidSchedule :: Gr a -> [Step] -> Bool
isValidSchedule gr steps = go steps IS.empty
  where
    go (Compute n : steps) live = all (`IS.member` live) (preList gr n) && go steps (IS.insert n live)
    go (Free n : steps) live = IS.member n live && go steps (IS.delete n live)
    go [] live = True

main = defaultMain $ testGroup "Tests" [
  testGraph,
  testBalanced,
  testTreeWidth,
  testGroup "TWRemat" [
      testProperty "produces valid schedule" $ \(DagOf (gr :: Gr ())) ->
          let t = last (nodes gr)
          in isValidSchedule gr (remat gr (IS.fromList [t])),
      testProperty "produces valid schedule x2" $ \(DagOf (gr :: Gr ())) ->
          let t = take 2 (nodes gr)
          in isValidSchedule gr (remat gr (IS.fromList t))
      ]
  ]
