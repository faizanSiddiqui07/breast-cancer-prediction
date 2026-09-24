const FEATURES = [
  "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
  "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
  "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
  "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
  "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
  "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
];

const GROUPS = {
  meanFields: FEATURES.slice(0, 10),
  seFields: FEATURES.slice(10, 20),
  worstFields: FEATURES.slice(20, 30)
};

const DEMO = {
  radius_mean: 14.0, texture_mean: 20.0, perimeter_mean: 92.0, area_mean: 600.0,
  smoothness_mean: 0.10, compactness_mean: 0.10, concavity_mean: 0.08,
  "concave points_mean": 0.05, symmetry_mean: 0.18, fractal_dimension_mean: 0.06,
  radius_se: 0.5, texture_se: 1.2, perimeter_se: 3.5, area_se: 40.0,
  smoothness_se: 0.007, compactness_se: 0.02, concavity_se: 0.02,
  "concave points_se": 0.01, symmetry_se: 0.02, fractal_dimension_se: 0.003,
  radius_worst: 16.0, texture_worst: 25.0, perimeter_worst: 105.0, area_worst: 800.0,
  smoothness_worst: 0.13, compactness_worst: 0.20, concavity_worst: 0.20,
  "concave points_worst": 0.10, symmetry_worst: 0.25, fractal_dimension_worst: 0.08
};

const labels = {
  radius:"Radius", texture:"Texture", perimeter:"Perimeter", area:"Area",
  smoothness:"Smoothness", compactness:"Compactness", concavity:"Concavity",
  "concave points":"Concave Points", symmetry:"Symmetry", fractal_dimension:"Fractal Dimension"
};

function labelFor(feature) {
  const suffix = feature.includes("_mean") ? "mean" :
                 feature.includes("_se") ? "SE" : "worst";
  const base = feature.replace("_mean","").replace("_se","").replace("_worst","");
  return `${labels[base] || base} <span class="tiny">${suffix}</span>`;
}

function createFields() {
  Object.entries(GROUPS).forEach(([containerId, features]) => {
    const container = document.getElementById(containerId);
    container.innerHTML = features.map(f => `
      <div class="field">
        <label for="${f}">${labelFor(f)}</label>
        <input id="${f}" name="${f}" type="number" min="0" step="any" placeholder="0.00" required />
      </div>
    `).join("");
  });
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predict(values) {
  if (
    typeof MODEL_CONFIG === "undefined" ||
    !MODEL_CONFIG.weights ||
    !MODEL_CONFIG.means ||
    !MODEL_CONFIG.scales
  ) {
    throw new Error("Model parameters are not connected yet. Add weights, scaler means and scaler scales to model-config.js.");
  }

  const { weights, bias, means, scales, threshold = 0.25 } = MODEL_CONFIG;

  if (weights.length !== FEATURES.length || means.length !== FEATURES.length || scales.length !== FEATURES.length) {
    throw new Error("Model configuration must contain exactly 30 weights, means and scales.");
  }

  let z = Number(bias || 0);

  for (let i = 0; i < FEATURES.length; i++) {
    const scaled = (values[i] - means[i]) / scales[i];
    z += scaled * weights[i];
  }

  return { probability: sigmoid(z), threshold };
}

function showResult(probability, threshold) {
  const malignant = probability >= threshold;
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultState").classList.remove("hidden");

  const badge = document.getElementById("resultBadge");
  badge.className = `result-badge ${malignant ? "malignant" : "benign"}`;
  badge.textContent = malignant ? "Malignant classification" : "Benign classification";

  document.getElementById("probabilityValue").textContent =
    `${(probability * 100).toFixed(2)}%`;

  document.getElementById("meterFill").style.width =
    `${Math.min(100, probability * 100)}%`;

  document.getElementById("resultMessage").textContent = malignant
    ? "The model classified this sample as malignant because its predicted probability is above the selected 25% threshold. This is an ML classification result, not a medical diagnosis."
    : "The model classified this sample as benign because its predicted probability is below the selected 25% threshold. This is an ML classification result, not a medical diagnosis.";
}

createFields();

document.getElementById("demoBtn").addEventListener("click", () => {
  FEATURES.forEach(f => {
    document.getElementById(f).value = DEMO[f];
  });
});

document.getElementById("clearBtn").addEventListener("click", () => {
  document.getElementById("predictionForm").reset();
  document.getElementById("emptyState").classList.remove("hidden");
  document.getElementById("resultState").classList.add("hidden");
});

document.getElementById("againBtn").addEventListener("click", () => {
  document.getElementById("resultState").classList.add("hidden");
  document.getElementById("emptyState").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

document.getElementById("predictionForm").addEventListener("submit", (event) => {
  event.preventDefault();

  const values = FEATURES.map(f => Number(document.getElementById(f).value));

  try {
    const result = predict(values);
    showResult(result.probability, result.threshold);
  } catch (error) {
    alert(error.message);
  }
});
