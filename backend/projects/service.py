from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.projects.models import Project
from backend.projects.schemas import ProjectCreate, ProjectUpdate


class ProjectNotFoundError(Exception):
    pass


def list_projects(db: Session) -> list[Project]:
    stmt = select(Project).order_by(Project.created_at.desc())
    return list(db.scalars(stmt))


def get_project(db: Session, project_id: int) -> Project | None:
    return db.get(Project, project_id)


def create_project(db: Session, user: User, payload: ProjectCreate) -> Project:
    project = Project(
        name=payload.name,
        summary=payload.summary,
        description=payload.description,
        requirements=payload.requirements,
        created_by=user.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def update_project(db: Session, project: Project, payload: ProjectUpdate) -> Project:
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project: Project) -> None:
    db.delete(project)
    db.commit()
