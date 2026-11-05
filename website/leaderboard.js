(() => {
  const sourceBody = document.querySelector("#full-results-body");
  const overallSystemSelect = document.querySelector("#overall-system");
  const datasetSelect = document.querySelector("#leaderboard-dataset");
  const systemSelect = document.querySelector("#leaderboard-system");
  const metricSelect = document.querySelector("#leaderboard-metric");
  const metricLabel = metricSelect?.closest("label");
  const tableHead = document.querySelector("#leaderboard-head");
  const resultsBody = document.querySelector("#leaderboard-body");
  if (!sourceBody || !overallSystemSelect || !datasetSelect || !systemSelect || !metricSelect || !tableHead || !resultsBody) return;

  const rows = [];
  let activeStage = "";
  sourceBody.querySelectorAll("tr").forEach((row) => {
    if (row.classList.contains("result-group")) {
      activeStage = row.textContent.match(/Stage\s+(\d)/)?.[1] ?? "";
      return;
    }
    const cells = [...row.cells].map((cell) => cell.textContent.trim());
    if (cells.length !== 7 || !activeStage) return;
    rows.push({ stage: activeStage, model: cells[0], task: cells[1], metric: cells[2], values: cells.slice(3) });
  });

  const tasks = [];
  const taskKeys = new Set();
  rows.forEach((row) => {
    const key = `${row.stage}::${row.task}`;
    if (!taskKeys.has(key)) {
      taskKeys.add(key);
      tasks.push({ key, stage: row.stage, name: row.task, metric: row.metric });
    }
  });

  const splitValues = (text) => text.split(/\s*\/\s*/).filter(Boolean);
  const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  const taskRows = new Map(tasks.map((task) => [task.key, new Map()]));
  rows.forEach((row) => taskRows.get(`${row.stage}::${row.task}`)?.set(row.model, row));
  const models = [...new Set(rows.map((row) => row.model))];
  const modelDisplayName = (model, selector) => {
    const configuration = selector.selectedOptions[0]?.textContent.trim() ?? "";
    return configuration === "Base" ? `${model} (Base)` : `${model} + ${configuration}`;
  };
  const metricNames = (metric) => {
    if (/^Bin\/Micro\/Macro$/i.test(metric)) return ["Binary F1", "Micro F1", "Macro F1"];
    return splitValues(metric).map((part) => {
      if (/^Score=2$/i.test(part)) return "Answer score (2)";
      if (/^(Acc|Accuracy)$/i.test(part)) return "Accuracy";
      if (/^Evidence$/i.test(part)) return "Evidence correctness";
      if (/^Correctness$/i.test(part)) return "Correctness";
      return part;
    });
  };

  tasks.forEach((task) => datasetSelect.add(new Option(`Stage ${task.stage} · ${task.name}`, task.key)));
  const taskForKey = (key) => tasks.find((task) => task.key === key) ?? tasks[0];

  const renderOverall = () => {
    const body = document.querySelector("#overall-ranking-body");
    if (!body) return;
    const systemIndex = Number(overallSystemSelect.value) - 3;
    const stages = ["1", "2", "3"].filter((stage) => tasks.some((task) => task.stage === stage));
    const stageScores = new Map(models.map((model) => [model, new Map()]));

    tasks.forEach((task) => {
      models.forEach((model) => {
        const row = taskRows.get(task.key)?.get(model);
        const primaryValue = splitValues(row?.values[systemIndex] ?? "")[0] ?? "";
        const score = Number.parseFloat(primaryValue.replace(/[^\d.-]/g, ""));
        if (!Number.isFinite(score)) return;
        const modelStages = stageScores.get(model);
        modelStages.get(task.stage) ?? modelStages.set(task.stage, []);
        modelStages.get(task.stage).push(score);
      });
    });

    const standings = models.map((model) => {
      const stageAverages = stages.map((stage) => mean(stageScores.get(model).get(stage) ?? []));
      const overallScore = mean(stageAverages.filter((value) => value !== null));
      return { model, stageAverages, overallScore };
    }).sort((a, b) => (b.overallScore ?? -Infinity) - (a.overallScore ?? -Infinity) || a.model.localeCompare(b.model));

    body.replaceChildren(...standings.map(({ model, stageAverages, overallScore }, rankIndex) => {
      const row = document.createElement("tr");
      const values = [String(rankIndex + 1).padStart(2, "0"), modelDisplayName(model, overallSystemSelect), overallScore?.toFixed(1) ?? "—", ...stages.map((_, index) => stageAverages[index]?.toFixed(1) ?? "—")];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        if (index === 0) cell.className = "overall-rank";
        if (index === 1) cell.className = "overall-model";
        if (index === 2) cell.className = "overall-score";
        cell.textContent = value;
        row.append(cell);
      });
      return row;
    }));
    document.querySelector("#overall-ranking-system").textContent = overallSystemSelect.selectedOptions[0]?.textContent ?? "Selected system";
  };

  const updateMetrics = () => {
    const selected = taskForKey(datasetSelect.value);
    const names = metricNames(selected?.metric ?? "");
    metricSelect.replaceChildren(...names.map((name, index) => new Option(name, String(index))));
    if (metricLabel) metricLabel.hidden = names.length < 2;
    render();
  };

  const render = () => {
    const selectedTask = taskForKey(datasetSelect.value);
    const metricIndex = Number(metricSelect.value || 0);
    const systemIndex = Number(systemSelect.value) - 3;
    const orderedStages = ["1", "2", "3"].filter((stage) => tasks.some((task) => task.stage === stage));
    const headerTop = document.createElement("tr");
    const rankHeading = document.createElement("th");
    rankHeading.scope = "col";
    rankHeading.rowSpan = 2;
    rankHeading.textContent = "Rank";
    const modelHeading = document.createElement("th");
    modelHeading.scope = "col";
    modelHeading.rowSpan = 2;
    modelHeading.textContent = "Model";
    headerTop.append(rankHeading, modelHeading);

    orderedStages.forEach((stage) => {
      const th = document.createElement("th");
      th.scope = "colgroup";
      th.colSpan = tasks.filter((task) => task.stage === stage).length;
      th.textContent = `Stage ${stage}`;
      headerTop.append(th);
    });
    const headerTasks = document.createElement("tr");
    tasks.forEach((task) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = task.name;
      th.dataset.metrics = metricNames(task.metric).join(" / ");
      if (task.key === selectedTask?.key) th.classList.add("rank-source");
      headerTasks.append(th);
    });
    tableHead.replaceChildren(headerTop, headerTasks);

    const getRankScore = (model) => {
      const row = taskRows.get(selectedTask?.key)?.get(model);
      const scorePart = splitValues(row?.values[systemIndex] ?? "")[metricIndex] ?? "";
      const score = Number.parseFloat(scorePart.replace(/[^\d.-]/g, ""));
      return Number.isFinite(score) ? score : -1;
    };
    models.sort((a, b) => getRankScore(b) - getRankScore(a) || a.localeCompare(b));

    resultsBody.replaceChildren(...models.map((model, rankIndex) => {
      const tr = document.createElement("tr");
      const rank = document.createElement("td");
      rank.className = "leaderboard-rank";
      rank.textContent = String(rankIndex + 1).padStart(2, "0");
      const modelCell = document.createElement("td");
      modelCell.className = "leaderboard-model";
      modelCell.textContent = modelDisplayName(model, systemSelect);
      tr.append(rank, modelCell);

      tasks.forEach((task) => {
        const td = document.createElement("td");
        td.className = "leaderboard-score-cell";
        td.dataset.task = `Stage ${task.stage} · ${task.name}`;
        if (task.key === selectedTask?.key) td.classList.add("rank-source");
        const row = taskRows.get(task.key)?.get(model);
        const scoreParts = splitValues(row?.values[systemIndex] ?? "—");
        scoreParts.forEach((value, index) => {
          const score = document.createElement("strong");
          score.className = "leaderboard-cell-score";
          score.textContent = value;
          score.setAttribute("aria-label", `${metricNames(task.metric)[index] ?? `Metric ${index + 1}`}: ${value}`);
          td.append(score);
        });
        tr.append(td);
      });
      return tr;
    }));

    document.querySelector("#leaderboard-context").textContent = "CellularS2-Bench · All datasets";
    document.querySelector("#leaderboard-model-count").textContent = `${models.length} models · ranked by ${selectedTask ? `Stage ${selectedTask.stage} · ${selectedTask.name}` : "dataset"}`;
  };

  datasetSelect.addEventListener("change", updateMetrics);
  systemSelect.addEventListener("change", render);
  metricSelect.addEventListener("change", render);
  overallSystemSelect.addEventListener("change", renderOverall);
  renderOverall();
  updateMetrics();
})();
