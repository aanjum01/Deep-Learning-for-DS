import numpy as np

from scripts.preprocess_mind import build

# N1,N2 = a; N3,N4 = b; N5 = c; N6 = z (z never shown). ctx_cats = [a, b, c, z]; arms = [a, b, c]
NEWS = "N1\ta\nN2\ta\nN3\tb\nN4\tb\nN5\tc\nN6\tz\n"


def write(root, train, dev):
    for s, rows in [("train", train), ("dev", dev)]:
        (root / s).mkdir()
        (root / s / "news.tsv").write_text(NEWS)
        (root / s / "behaviors.tsv").write_text("".join("\t".join(r) + "\n" for r in rows))


def row(i, user, time, hist, imps):
    return (str(i), user, f"11/15/2019 {time}", hist, imps)


def test_avail_r_and_arms(tmp_path):
    write(tmp_path, [row(1, "U1", "10:00:00 AM", "", "N1-1 N2-0 N3-0 N4-1")],
          [row(1, "U2", "11:00:00 AM", "", "N3-0 N5-1")])
    o = build(tmp_path)
    assert list(o["ctx_cats"]) == ["a", "b", "c", "z"] and list(o["arms"]) == ["a", "b", "c"]
    np.testing.assert_array_equal(o["avail"], [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(o["R"], [[1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(o["split"], [0, 1])
    np.testing.assert_array_equal(o["impression_id"], [1, 1])


def test_sorted_and_duplicate_dropped(tmp_path):
    dup = row(5, "U1", "09:00:00 AM", "", "N1-1 N3-0")
    write(tmp_path, [row(1, "U1", "10:00:00 PM", "", "N1-1"), dup], [row(9, "U2", "08:00:00 AM", "", "N5-1"), dup])
    o = build(tmp_path)
    assert o["n_dup"] == 1
    assert list(o["time"]) == sorted(o["time"]) and len(o["time"]) == 3
    np.testing.assert_array_equal(o["impression_id"], [9, 5, 1])
    np.testing.assert_array_equal(o["split"], [1, 0, 0])  # kept the train copy of the duplicate


def test_history_update_no_leak(tmp_path):
    write(tmp_path, [row(1, "U1", "10:00:00 AM", "N3", "N1-1 N1-1 N5-1"),  # sorted order: 1, 2 (U2), 3
                     row(2, "U2", "10:30:00 AM", "", "N4-1"),
                     row(3, "U1", "11:00:00 AM", "N3", "N2-0")], [])
    X = build(tmp_path)["X"]
    # 1st U1: history {b:1} only; its own clicks (a,a,c) are absent
    np.testing.assert_allclose(X[0], [0, 1, 0, 0, 1])
    # U2: its history is empty and U1's clicks do not leak in
    np.testing.assert_allclose(X[1], [0, 0, 0, 0, 1])
    # 2nd U1: history b:1 + earlier clicks a:2, c:1 -> counts [2,1,1,0] / 4
    np.testing.assert_allclose(X[2], [.5, .25, .25, 0, 1])
    assert X.dtype == np.float32


def test_empty_history_is_zeros_and_bias(tmp_path):
    write(tmp_path, [row(1, "U1", "10:00:00 AM", "", "N1-1")], [])
    np.testing.assert_array_equal(build(tmp_path)["X"], [[0, 0, 0, 0, 1]])


def test_small_users_filter(tmp_path):
    write(tmp_path, [row(1, "U1", "10:00:00 AM", "", "N1-1"), row(2, "U2", "10:01:00 AM", "", "N3-1"),
                     row(3, "U3", "10:02:00 AM", "", "N5-1")], [])
    o = build(tmp_path, small_users={"U1", "U3"})
    assert o["n_filtered"] == 2 and len(o["X"]) == 1
    np.testing.assert_array_equal(o["impression_id"], [2])
