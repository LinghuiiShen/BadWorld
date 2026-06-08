import React from 'react';
import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';

marked.use(markedKatex({ throwOnError: false }));

export default class Overview extends React.Component {
  render() {
    const hasAbstract =
      typeof this.props.abstract === 'string' &&
      this.props.abstract.trim().length > 0;

    return (
      <div className="uk-section">
        {this.props.teaser && (
          <img
            src={`${this.props.teaser}`}
            className="uk-align-center uk-responsive-width"
            alt=""
          />
        )}

        {hasAbstract && (
          <>
            <h2 className="uk-text-bold uk-heading-line uk-text-center">
              <span>Abstract</span>
            </h2>
            <div
              dangerouslySetInnerHTML={{
                __html: marked.parse(this.props.abstract),
              }}
            />
          </>
        )}
      </div>
    );
  }
}
