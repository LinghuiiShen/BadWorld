import React from 'react';

const ProjectCard = ({ title, url }) => (
  <a
    href={url}
    target="_blank"
    rel="noopener noreferrer"
    className="related-project-pill"
  >
    <span>{title}</span>
    <span className="related-project-arrow">↗</span>
  </a>
);

export default class Projects extends React.Component {
  render() {
    if (!this.props.projects || this.props.projects.length === 0) {
      return null;
    }

    return (
      <div className="uk-section related-projects-section">
        <style>{`
          .related-projects-section {
            padding-top: 25px;
          }

          .related-projects-list {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 22px;
          }

          .related-project-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 22px;
            border-radius: 999px;
            border: 1px solid rgba(211,47,47,0.22);
            background: rgba(211,47,47,0.055);
            color: #111;
            font-weight: 800;
            font-size: 1.05em;
            text-decoration: none !important;
            transition: all 0.18s ease;
          }

          .related-project-pill:hover {
            color: #d32f2f;
            border-color: rgba(211,47,47,0.45);
            background: rgba(211,47,47,0.095);
            transform: translateY(-1px);
          }

          .related-project-arrow {
            color: #d32f2f;
            font-weight: 900;
          }
        `}</style>

        <h2 className="uk-heading-line uk-text-center uk-text-bold">
          <span>Relevant Projects</span>
        </h2>

        <div className="related-projects-list">
          {this.props.projects.map((project, idx) => (
            <ProjectCard
              key={idx}
              title={project.title}
              url={project.url}
            />
          ))}
        </div>
      </div>
    );
  }
}