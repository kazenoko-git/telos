const $ = selector => document.querySelector(selector);
const notice = message => { $("#notice").textContent = message; };

async function api(path, method = "GET", data) {
  const response = await fetch("/api/" + path, {
    method, credentials: "same-origin", headers: { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || "The request failed.");
  return value;
}

async function act(fn) {
  notice("");
  try { await fn(); } catch (error) { notice(error.message); }
}

function node(tag, value, className) {
  const element = document.createElement(tag);
  if (value !== undefined) element.textContent = value;
  if (className) element.className = className;
  return element;
}

function button(label, fn) {
  const element = node("button", label, "secondary");
  element.type = "button";
  element.addEventListener("click", () => act(fn));
  return element;
}

function reveal(value) {
  $("#secret-value").value = value;
  $("#secret").showModal();
}

async function refresh() {
  const owner = await api("me");
  $("#auth").hidden = true;
  $("#dashboard").hidden = false;
  $("#logout").hidden = false;
  $("#email").textContent = owner.email;
  const [keys, hosts] = await Promise.all([api("keys"), api("hosts")]);
  $("#keys").replaceChildren();
  for (const key of keys) {
    const row = node("div", undefined, "item");
    row.append(node("h3", key.name), node("span", key.prefix + "…", "muted"));
    if (key.revoked_at) row.append(node("p", "Revoked", "muted"));
    else {
      const actions = node("div", undefined, "actions");
      actions.append(
        button("Rotate", async () => {
          if (!confirm("Rotate this key? Its current value will stop authorizing new requests.")) return;
          const result = await api("keys/" + key.id + "/rotate", "POST", {});
          reveal(result.key);
          await refresh();
        }),
        button("Revoke", async () => {
          if (!confirm("Revoke this key? Clients using it will need another key.")) return;
          await api("keys/" + key.id, "DELETE");
          await refresh();
        }),
      );
      row.append(actions);
    }
    $("#keys").append(row);
  }
  if (!keys.length) $("#keys").append(node("p", "Create a key to connect your API clients."));
  $("#hosts").replaceChildren();
  for (const host of hosts) {
    const row = node("div", undefined, "item");
    row.append(node("h3", host.model), node("code", "https://" + host.slug + ".tls.wingit.tech/v1"));
    row.append(node("p", host.last_seen > Date.now()/1000 - 90 ? "Online" : "Offline", "muted"));
    const command = 'telos serve --checkpoint "<path>" --model-name "' + host.model + '" --tunnel ' + host.slug;
    row.append(node("code", command));
    const share = node("label", undefined, "share");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(host.shared);
    checkbox.addEventListener("change", () => act(async () => {
      checkbox.disabled = true;
      try { await api("hosts/" + host.slug, "PATCH", { shared: checkbox.checked }); }
      catch (error) { checkbox.checked = !checkbox.checked; throw error; }
      finally { checkbox.disabled = false; }
    }));
    share.append(checkbox, document.createTextNode("Offer this model to the shared pool"));
    row.append(share);
    $("#hosts").append(row);
  }
  if (!hosts.length) $("#hosts").append(node("p", "Reserve a subdomain, then connect your model with telos serve."));
}

async function authenticate(action) {
  const form = $("#login");
  if (!form.reportValidity()) return;
  const data = Object.fromEntries(new FormData(form));
  await api(action, "POST", data);
  form.reset();
  await refresh();
}
$("#login").addEventListener("submit", event => { event.preventDefault(); act(() => authenticate("login")); });
$("#signup").addEventListener("click", () => act(() => authenticate("signup")));
$("#logout").addEventListener("click", () => act(async () => {
  await api("logout", "POST", {});
  location.reload();
}));
$("#new-key").addEventListener("submit", event => {
  event.preventDefault();
  act(async () => {
    const result = await api("keys", "POST", Object.fromEntries(new FormData(event.target)));
    event.target.reset();
    reveal(result.key);
    await refresh();
  });
});
$("#new-host").addEventListener("submit", event => {
  event.preventDefault();
  act(async () => {
    await api("hosts", "POST", Object.fromEntries(new FormData(event.target)));
    event.target.reset();
    await refresh();
  });
});
$("#copy-secret").addEventListener("click", () => act(async () => {
  await navigator.clipboard.writeText($("#secret-value").value);
  $("#copy-secret").textContent = "Copied";
}));
$("#close-secret").addEventListener("click", () => $("#secret").close());
$("#secret").addEventListener("close", () => {
  $("#secret-value").value = "";
  $("#copy-secret").textContent = "Copy key";
});
refresh().catch(() => {
  $("#auth").hidden = false;
  $("#dashboard").hidden = true;
});
