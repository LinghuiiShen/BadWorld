import React from 'react';
import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';

marked.use(markedKatex({ throwOnError: false }));

export default class Overview extends React.Component {
  render() {
    const hasAbstract =
      typeof this.props.abstract === 'string' &&
      this.props.abstract.trim().length > 0;

    const isTeaserVideo =
      typeof this.props.teaser === 'string' &&
      /\.(mp4|webm|ogg)$/i.test(this.props.teaser);

    return (
      <section className="uk-section">
        <div className="uk-container uk-container-small">
          {this.props.teaser &&
            (isTeaserVideo ? (
              <video
                src={this.props.teaser}
                className="uk-width-1-1"
                autoPlay
                muted
                loop
                playsInline
                preload="auto"
                controls
              />
            ) : (
              <img src={this.props.teaser} className="uk-width-1-1" />
            ))}

          {hasAbstract && (
            <>
              <h2 className="uk-h2 uk-text-primary uk-text-bold">
                Abstract
              </h2>
              <div
                dangerouslySetInnerHTML={{
                  __html: marked.parse(this.props.abstract),
                }}
              />
            </>
          )}
        </div>
      </section>
    );
  }
}