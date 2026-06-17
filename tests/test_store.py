import pytest

from src.api.store import DraftNotFoundError, InMemoryScoreRepository, StaleRevisionError


def test_in_memory_store_manages_draft_lifecycle():
    repository = InMemoryScoreRepository[dict[str, object]]()

    created = repository.create_score({"title": "Prelude", "notes": ["C4"]})
    assert created.score_id == "score-1"
    assert created.revision == 1
    assert created.score == {"title": "Prelude", "notes": ["C4"]}

    draft = repository.create_draft(created.score_id, base_revision=created.revision)
    assert draft.draft_id == "draft-1"
    assert draft.score_id == created.score_id
    assert draft.base_revision == 1
    assert draft.score == created.score

    updated = repository.save_draft(
        draft.draft_id,
        {"title": "Prelude in C", "notes": ["C4", "E4"]},
    )
    assert updated.score == {"title": "Prelude in C", "notes": ["C4", "E4"]}

    committed = repository.commit_draft(draft.draft_id)
    assert committed.score_id == created.score_id
    assert committed.revision == 2
    assert committed.score == {"title": "Prelude in C", "notes": ["C4", "E4"]}
    assert repository.get_score(created.score_id) == committed

    throwaway = repository.create_draft(created.score_id)
    repository.discard_draft(throwaway.draft_id)
    with pytest.raises(DraftNotFoundError):
        repository.get_draft(throwaway.draft_id)


def test_in_memory_store_rejects_stale_commits():
    repository = InMemoryScoreRepository[dict[str, object]]()
    created = repository.create_score({"title": "Allemande"})

    first_draft = repository.create_draft(created.score_id, base_revision=created.revision)
    second_draft = repository.create_draft(created.score_id, base_revision=created.revision)

    repository.save_draft(first_draft.draft_id, {"title": "Allemande", "tempo": "grave"})
    repository.save_draft(second_draft.draft_id, {"title": "Allemande", "tempo": "vivace"})

    committed = repository.commit_draft(first_draft.draft_id)
    assert committed.revision == 2
    assert committed.score == {"title": "Allemande", "tempo": "grave"}

    with pytest.raises(StaleRevisionError):
        repository.commit_draft(second_draft.draft_id)

    with pytest.raises(StaleRevisionError):
        repository.create_draft(created.score_id, base_revision=1)
