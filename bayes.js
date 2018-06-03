// This is a simple JS version of the cookie
// problem from Allen Downey's tutorial
// "Bayesian statistics made simple":
//
//   https://www.youtube.com/watch?v=TpgiFIGXcT4
//
// It can also be found in the tutorial's GitHub
// repository:
//
//   https://github.com/AllenDowney/BayesMadeSimple

const VANILLA = 'vanilla';
const CHOCOLATE = 'chocolate';

const VANILLA_PROBS = {
  bowl1: 0.75,
  bowl2: 0.5,
};

const HYPOTHESES = Object.keys(VANILLA_PROBS);

// Return the sum of the given probability map.
function sum(probs) {
  return HYPOTHESES.reduce((total, h) => total + probs[h], 0);
}

// Return P(D|H), i.e. the probability of the data given a
// hypothesis.
function likelihood(data, hypothesis) {
  const vanillaProb = VANILLA_PROBS[hypothesis];
  return data === VANILLA ? vanillaProb : 1 - vanillaProb;
}

// Return P(D), i.e. the probability of the given data,
// regardless of hypothesis.
function dataProb(data) {
  const vanillaProb = sum(VANILLA_PROBS) / HYPOTHESES.length;
  return data === VANILLA ? vanillaProb : 1 - vanillaProb;
}

// Return the P(H|D), i.e. the probability of a particular
// hypothesis given a data point.
//
// By Bayes' theorem, this is: P(H|D) = ( P(H) * P(D|H) ) / P(D)
function hypothesisProb(priors, hypothesis, data) {
  return (priors[hypothesis] * likelihood(data, hypothesis)) / dataProb(data);
}

// Make a probability map, given a function that maps a hypothesis to a
// probability.
function makeProbs(fn) {
  const result = {};

  HYPOTHESES.forEach(hypothesis => {
    result[hypothesis] = fn(hypothesis);
  });

  return result;
}

// Normalize a probability map by dividing every probability by the
// sum of probabilities.
function normalize(probs) {
  const total = sum(probs);
  return makeProbs(hypothesis => probs[hypothesis] / total);
}

// Perform one step of the diachronic interpretation of Bayes' theorem.
function updatePriors(priors, data) {
  return normalize(makeProbs(hypothesis => {
    return hypothesisProb(priors, hypothesis, data)
  }));
}

// Return a human-readable string interpretation of a probability map.
function strProbs(probs) {
  return HYPOTHESES.map(hypothesis => {
    return `${hypothesis}=${probs[hypothesis].toFixed(2)}`;
  }).join(', ');
}

// Given an array of data points, run the diachronic interpretation of Bayes'
// theorem through it and return the final probability map.
//
// The initial priors are a probability map where every hypothesis is
// equally likely.
function runBayes(allData) {
  let priors = makeProbs(hypothesis => 1 / HYPOTHESES.length);

  allData.forEach(data => {
    priors = updatePriors(priors, data);
    console.log(`got ${data}, new priors: ${strProbs(priors)}`);
  });
}

// Generate sample data for the given hypothesis.
function generateData(hypothesis, count) {
  const data = [];
  const vanillaProb = likelihood(VANILLA, hypothesis);

  for (let i = 0; i < count; i++) {
    const val = Math.random();
    data.push(val < vanillaProb ? VANILLA : CHOCOLATE);
  }

  return data;
}

runBayes(generateData('bowl1', 20));
