// Keep obsolete numerical work out of the queue. Search steps return to the
// event loop so new requests and cancellation messages can be observed.
function yieldWorkerTask() {
  return new Promise(resolve => {
    if (typeof MessageChannel === "undefined") { setTimeout(resolve, 0); return; }
    // Nested timers are clamped to 4 ms in browsers. A posted message yields
    // a task without adding that fixed delay to every numerical search row.
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      channel.port2.close();
      resolve();
    };
    channel.port2.postMessage(null);
  });
}

export class LatestRequestQueue {
  constructor({execute, startSearch, post, yieldTask = yieldWorkerTask}) {
    this.execute = execute;
    this.startSearch = startSearch;
    this.post = post;
    this.yieldTask = yieldTask;
    this.pending = [];
    this.activeTask = null;
    this.search = null;
    this.running = false;
    this.idleWaiters = [];
  }

  enqueue(message) {
    if (message.type === "cancel") {
      this.cancel(message.action);
      return;
    }
    const action = message.request?.action;
    if (action === "render" || action === "near_search") this.cancel(action);
    this.pending.push({...message, cancelled: false, settled: false});
    void this.pump();
  }

  cancel(action) {
    const matches = job => !action || job.request.action === action;
    for (const job of this.pending) if (matches(job)) this.cancelJob(job);
    this.pending = this.pending.filter(job => !job.cancelled);
    if (this.activeTask && matches(this.activeTask)) this.cancelJob(this.activeTask);
    if (this.search && matches(this.search.job)) this.cancelJob(this.search.job);
  }

  cancelJob(job) {
    job.cancelled = true;
    if (job.settled) return;
    job.settled = true;
    this.post({type: "error", id: job.id, name: "AbortError", message: "Request superseded"});
  }

  finish(job, output) {
    if (job.settled) return;
    this.post({type: "result", id: job.id, result: output.result}, output.transfer || []);
    job.settled = true;
  }

  fail(job, error) {
    if (job.settled) return;
    job.settled = true;
    this.post({type: "error", id: job.id, name: error.name || "Error", message: error.message || String(error)});
  }

  whenIdle() {
    return this.running || this.pending.length || this.search
      ? new Promise(resolve => this.idleWaiters.push(resolve)) : Promise.resolve();
  }

  async pump() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.pending.length || this.search) {
        if (this.search?.job.cancelled) {
          this.search.handle.dispose();
          this.search = null;
        }
        // Render, picking and session actions take precedence over the next
        // search block; a long search cannot hold up these user interactions.
        const index = this.pending.findIndex(job => job.request.action !== "near_search");
        const job = index >= 0 ? this.pending.splice(index, 1)[0]
          : !this.search ? this.pending.shift() : null;
        if (job) {
          this.activeTask = job;
          try {
            if (job.request.action === "near_search") {
              const handle = await this.startSearch(job.request, () => job.cancelled);
              if (job.cancelled) handle?.dispose();
              else this.search = {job, handle};
            } else {
              const output = await this.execute(job.request, () => job.cancelled);
              if (!job.cancelled) this.finish(job, output);
            }
          } catch (error) { this.fail(job, error); }
          finally { this.activeTask = null; }
        } else if (this.search) {
          const {job: searchJob, handle} = this.search;
          try {
            if (await handle.step()) {
              if (!searchJob.cancelled) this.finish(searchJob, await handle.finish());
              handle.dispose();
              this.search = null;
            }
          } catch (error) {
            this.fail(searchJob, error);
            handle.dispose();
            this.search = null;
          }
        }
        // A resolved Promise alone would only yield to microtasks, starving
        // Worker message events. Yield a task after every bounded operation.
        await this.yieldTask();
      }
    } finally {
      this.running = false;
      for (const resolve of this.idleWaiters.splice(0)) resolve();
    }
  }
}
